//! Translate an inbound Responses API request into agentix's internal
//! representation. Resolves `previous_response_id` against the session store
//! to prepend prior turn items. Reasoning items are preserved verbatim into
//! `Message::Assistant.provider_data["openai_responses_items"]` so signatures
//! / encrypted_content round-trip when the upstream is OpenAI.

use std::sync::Arc;

use serde_json::Value;

use crate::raw::shared::{FunctionDefinition, ToolDefinition, ToolKind};
use crate::request::{
    Content, DocumentContent, DocumentData, ImageContent, ImageData, Message, ReasoningEffort,
    ToolCall, ToolChoice, UserContent,
};
use crate::server::translated::Translated;

use super::error::OpenAIError;
use super::store::SessionStore;
use super::wire::{self, MessageContent, MessageContentPart, ResponsesRequest, TypedInputItem};

const PASSTHROUGH_KEYS: &[&str] = &[
    "top_p",
    "top_logprobs",
    "user",
    "metadata",
    "service_tier",
    "stop",
];

pub struct InboundContext {
    pub store: Arc<SessionStore>,
}

#[derive(Debug)]
pub struct PreparedRequest {
    pub translated: Translated,
    /// Verbatim items as they will be persisted to the session store after
    /// the upstream produces output. Each entry is a wire JSON value already
    /// flattened to the request shape (resolved input + transparent values).
    pub stored_input_items: Vec<Value>,
    /// `previous_response_id` from the request — used as the new entry's
    /// parent in the session store on commit.
    pub parent_id: Option<String>,
    /// Echo back to the client in the non-streaming response object.
    pub store_requested: bool,
    pub include: Vec<String>,
    pub reasoning_summary: Option<String>,
}

pub fn translate(
    req: ResponsesRequest,
    ctx: &InboundContext,
) -> Result<PreparedRequest, OpenAIError> {
    let stream = req.stream.unwrap_or(false);
    let parent_id = req.previous_response_id.clone();
    let store_requested = req.store;
    let include = req.include.clone().unwrap_or_default();
    let reasoning_summary = req.reasoning.as_ref().and_then(|r| r.summary.clone());

    // Resolve previous_response_id chain.
    let mut all_items: Vec<Value> = Vec::new();
    if let Some(prev) = &parent_id {
        match ctx.store.resolve(prev) {
            Some(items) => all_items.extend(items),
            None => {
                return Err(OpenAIError::invalid_request(format!(
                    "previous_response_id {prev} not found (expired or never stored)"
                )));
            }
        }
    }

    // Append this request's input.
    match req.input {
        wire::InputField::Text(s) => {
            all_items.push(serde_json::json!({
                "type": "message",
                "role": "user",
                "content": s,
            }));
        }
        wire::InputField::Items(v) => all_items.extend(v),
    }

    // Now translate the unified item list into agentix `Vec<Message>`.
    let messages = items_to_messages(&all_items)?;

    // Tools — Responses uses flat shape; reject non-function tool types
    // (web_search/file_search/computer_use) since agentix has no equivalent.
    let mut translated_tools: Vec<ToolDefinition> = Vec::new();
    for t in req.tools.unwrap_or_default() {
        if t.kind != "function" {
            return Err(OpenAIError::invalid_request(format!(
                "tool type `{}` not supported by agentix proxy (only function tools are forwarded)",
                t.kind
            )));
        }
        let name = t.name.ok_or_else(|| {
            OpenAIError::invalid_request("function tool missing required `name` field")
        })?;
        translated_tools.push(ToolDefinition {
            kind: ToolKind::Function,
            function: FunctionDefinition {
                name,
                description: t.description,
                parameters: t.parameters.unwrap_or(Value::Null),
                strict: t.strict,
            },
        });
    }

    let tool_choice = req.tool_choice.map(translate_tool_choice).transpose()?;
    let reasoning_effort = req
        .reasoning
        .as_ref()
        .and_then(|r| r.effort.as_deref())
        .and_then(parse_effort);

    let mut extra_body = serde_json::Map::new();
    for (k, v) in req.extra {
        if PASSTHROUGH_KEYS.contains(&k.as_str()) {
            extra_body.insert(k, v);
        }
    }

    let translated = Translated {
        system_prompt: req.instructions.filter(|s| !s.is_empty()),
        model_from_client: req.model,
        max_tokens: req.max_output_tokens.unwrap_or(4096),
        messages,
        tools: translated_tools,
        tool_choice,
        temperature: req.temperature,
        reasoning_effort,
        stream,
        extra_body,
    };

    Ok(PreparedRequest {
        translated,
        stored_input_items: all_items,
        parent_id,
        store_requested,
        include,
        reasoning_summary,
    })
}

fn items_to_messages(items: &[Value]) -> Result<Vec<Message>, OpenAIError> {
    let mut out: Vec<Message> = Vec::new();
    let mut pending_assistant: Option<AssistantBuilder> = None;

    for raw in items {
        // Try typed parse; on failure, treat as opaque (e.g. unsupported item
        // kinds in input — but those should have been rejected at the schema
        // layer if we cared).
        let typed: Result<TypedInputItem, _> = serde_json::from_value(raw.clone());
        match typed {
            Ok(TypedInputItem::Message { role, content, .. }) => {
                match role.as_str() {
                    "user" | "developer" => {
                        flush_assistant(&mut pending_assistant, &mut out);
                        out.push(Message::User(parts_from_content(content)));
                    }
                    "system" => {
                        // Older clients may emit system-role messages here
                        // even though Responses canonicalizes to top-level
                        // `instructions`. We don't have a place to put them
                        // mid-conversation; fold into a user message.
                        flush_assistant(&mut pending_assistant, &mut out);
                        out.push(Message::User(parts_from_content(content)));
                    }
                    "assistant" => {
                        let text = match &content {
                            MessageContent::Text(s) => s.clone(),
                            MessageContent::Parts(parts) => parts
                                .iter()
                                .filter_map(|p| match p {
                                    MessageContentPart::OutputText { text }
                                    | MessageContentPart::InputText { text } => {
                                        Some(text.as_str())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(""),
                        };
                        let asst = pending_assistant.get_or_insert_with(AssistantBuilder::default);
                        if !text.is_empty() {
                            if let Some(prev) = &mut asst.content {
                                prev.push_str(&text);
                            } else {
                                asst.content = Some(text);
                            }
                        }
                        asst.raw_items.push(raw.clone());
                    }
                    other => {
                        return Err(OpenAIError::invalid_request(format!(
                            "unexpected message role: {other}"
                        )));
                    }
                }
            }
            Ok(TypedInputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            }) => {
                let asst = pending_assistant.get_or_insert_with(AssistantBuilder::default);
                asst.tool_calls.push(ToolCall {
                    id: call_id,
                    name,
                    arguments,
                });
                asst.raw_items.push(raw.clone());
            }
            Ok(TypedInputItem::FunctionCallOutput {
                call_id, output, ..
            }) => {
                flush_assistant(&mut pending_assistant, &mut out);
                out.push(Message::ToolResult {
                    call_id,
                    content: vec![Content::text(output)],
                });
            }
            Ok(TypedInputItem::Reasoning { .. }) => {
                let asst = pending_assistant.get_or_insert_with(AssistantBuilder::default);
                asst.raw_items.push(raw.clone());
            }
            Err(_) => {
                // Unknown item type — preserve in pending assistant's raw
                // items so it round-trips into provider_data, in case the
                // upstream is OpenAI Responses and recognizes it.
                let asst = pending_assistant.get_or_insert_with(AssistantBuilder::default);
                asst.raw_items.push(raw.clone());
            }
        }
    }

    flush_assistant(&mut pending_assistant, &mut out);
    Ok(out)
}

#[derive(Default)]
struct AssistantBuilder {
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
    raw_items: Vec<Value>,
}

fn flush_assistant(b: &mut Option<AssistantBuilder>, out: &mut Vec<Message>) {
    if let Some(asst) = b.take() {
        if asst.content.is_none() && asst.tool_calls.is_empty() && asst.raw_items.is_empty() {
            return;
        }
        let provider_data = if asst.raw_items.is_empty() {
            None
        } else {
            Some(serde_json::json!({
                "openai_responses_items": asst.raw_items,
            }))
        };
        out.push(Message::Assistant {
            content: asst.content,
            reasoning: None,
            tool_calls: asst.tool_calls,
            provider_data,
        });
    }
}

fn parts_from_content(c: MessageContent) -> Vec<UserContent> {
    let parts = match c {
        MessageContent::Text(s) => {
            return if s.is_empty() {
                Vec::new()
            } else {
                vec![UserContent::Text { text: s }]
            };
        }
        MessageContent::Parts(p) => p,
    };
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        match p {
            MessageContentPart::InputText { text } | MessageContentPart::OutputText { text } => {
                if !text.is_empty() {
                    out.push(UserContent::Text { text });
                }
            }
            MessageContentPart::InputImage { image_url, .. } => {
                if let Some(url) = image_url {
                    let (data, mime) = parse_data_url_or_url(&url, ImageData::Url, ImageData::Base64);
                    out.push(UserContent::Image(ImageContent {
                        data,
                        mime_type: mime,
                    }));
                }
            }
            MessageContentPart::InputFile {
                file_data,
                file_url,
                filename,
            } => {
                let doc = if let Some(data) = file_data {
                    let (d, mime) = parse_data_url_or_url(&data, DocumentData::Url, DocumentData::Base64);
                    DocumentContent {
                        data: d,
                        mime_type: mime,
                        filename,
                    }
                } else if let Some(url) = file_url {
                    DocumentContent {
                        data: DocumentData::Url(url),
                        mime_type: String::new(),
                        filename,
                    }
                } else {
                    continue;
                };
                out.push(UserContent::Document(doc));
            }
            MessageContentPart::Unknown => {}
        }
    }
    out
}

fn parse_data_url_or_url<T>(
    url: &str,
    url_ctor: fn(String) -> T,
    base64_ctor: fn(String) -> T,
) -> (T, String) {
    if let Some(rest) = url.strip_prefix("data:")
        && let Some((meta, data)) = rest.split_once(',')
    {
        let mime = meta.split(';').next().unwrap_or("").to_string();
        return (base64_ctor(data.to_string()), mime);
    }
    (url_ctor(url.to_string()), String::new())
}

fn translate_tool_choice(tc: wire::ToolChoice) -> Result<ToolChoice, OpenAIError> {
    Ok(match tc {
        wire::ToolChoice::Named(s) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            other => {
                return Err(OpenAIError::invalid_request(format!(
                    "unknown tool_choice: {other}"
                )));
            }
        },
        wire::ToolChoice::Object(obj) => {
            if obj.kind != "function" {
                return Err(OpenAIError::invalid_request(format!(
                    "tool_choice type `{}` not supported",
                    obj.kind
                )));
            }
            let name = obj.name.ok_or_else(|| {
                OpenAIError::invalid_request("tool_choice with type=function missing `name`")
            })?;
            ToolChoice::Tool(name)
        }
    })
}

fn parse_effort(s: &str) -> Option<ReasoningEffort> {
    Some(match s {
        "none" => ReasoningEffort::None,
        "minimal" => ReasoningEffort::Minimal,
        "low" => ReasoningEffort::Low,
        "medium" => ReasoningEffort::Medium,
        "high" => ReasoningEffort::High,
        "xhigh" => ReasoningEffort::XHigh,
        "max" => ReasoningEffort::Max,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> InboundContext {
        InboundContext {
            store: Arc::new(SessionStore::default()),
        }
    }

    fn parse(body: serde_json::Value) -> PreparedRequest {
        let req: ResponsesRequest = serde_json::from_value(body).unwrap();
        translate(req, &ctx()).unwrap()
    }

    #[test]
    fn input_string_becomes_user_message() {
        let p = parse(json!({
            "model": "gpt-5",
            "input": "say hi",
        }));
        assert_eq!(p.translated.messages.len(), 1);
        assert!(matches!(&p.translated.messages[0], Message::User(_)));
    }

    #[test]
    fn input_items_typed_message_user() {
        let p = parse(json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hi"}
            ],
        }));
        assert_eq!(p.translated.messages.len(), 1);
    }

    #[test]
    fn instructions_become_system_prompt() {
        let p = parse(json!({
            "model": "gpt-5",
            "instructions": "be terse",
            "input": "hi",
        }));
        assert_eq!(p.translated.system_prompt.as_deref(), Some("be terse"));
    }

    #[test]
    fn function_call_then_function_call_output_split_into_assistant_and_toolresult() {
        let p = parse(json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "what's 7*8?"},
                {"type": "function_call", "call_id": "c1", "name": "mul", "arguments": "{\"a\":7,\"b\":8}"},
                {"type": "function_call_output", "call_id": "c1", "output": "56"},
            ],
        }));
        assert_eq!(p.translated.messages.len(), 3);
        assert!(matches!(&p.translated.messages[0], Message::User(_)));
        assert!(matches!(
            &p.translated.messages[1],
            Message::Assistant { tool_calls, .. } if tool_calls.len() == 1
        ));
        assert!(matches!(
            &p.translated.messages[2],
            Message::ToolResult { call_id, .. } if call_id == "c1"
        ));
    }

    #[test]
    fn reasoning_items_preserved_in_provider_data() {
        let p = parse(json!({
            "model": "gpt-5",
            "input": [
                {"type": "message", "role": "user", "content": "hi"},
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"},
                {"type": "function_call", "call_id": "c1", "name": "x", "arguments": "{}"},
            ],
        }));
        // Assistant message captures both reasoning + function_call as raw items.
        let asst = p.translated.messages.iter().find_map(|m| match m {
            Message::Assistant { provider_data, tool_calls, .. } => Some((provider_data.clone(), tool_calls.clone())),
            _ => None,
        }).unwrap();
        let pd = asst.0.unwrap();
        let items = pd.get("openai_responses_items").unwrap().as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["encrypted_content"], "ENC");
        assert_eq!(items[1]["type"], "function_call");
        assert_eq!(asst.1.len(), 1);
    }

    #[test]
    fn previous_response_id_resolves_chain() {
        let ctx = ctx();
        ctx.store.put(
            "r1".into(),
            vec![json!({"type": "message", "role": "user", "content": "first"})],
            None,
        );
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "previous_response_id": "r1",
            "input": "second",
        })).unwrap();
        let p = translate(req, &ctx).unwrap();
        assert_eq!(p.parent_id.as_deref(), Some("r1"));
        // Two user messages: "first" (resolved) + "second" (new input).
        let users: Vec<&Message> = p
            .translated
            .messages
            .iter()
            .filter(|m| matches!(m, Message::User(_)))
            .collect();
        assert_eq!(users.len(), 2);
    }

    #[test]
    fn unknown_previous_response_id_errors() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "previous_response_id": "missing",
            "input": "hi",
        })).unwrap();
        let err = translate(req, &ctx()).unwrap_err();
        assert_eq!(err.kind, super::super::error::ErrorKind::InvalidRequest);
    }

    #[test]
    fn tool_definition_flat_shape() {
        let p = parse(json!({
            "model": "gpt-5",
            "input": "hi",
            "tools": [{
                "type": "function",
                "name": "calc",
                "description": "compute",
                "parameters": {"type": "object"},
            }],
        }));
        assert_eq!(p.translated.tools.len(), 1);
        assert_eq!(p.translated.tools[0].function.name, "calc");
    }

    #[test]
    fn non_function_tool_rejected() {
        let req: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-5",
            "input": "hi",
            "tools": [{"type": "web_search"}],
        })).unwrap();
        let err = translate(req, &ctx()).unwrap_err();
        assert_eq!(err.kind, super::super::error::ErrorKind::InvalidRequest);
    }

    #[test]
    fn reasoning_effort_high_maps() {
        let p = parse(json!({
            "model": "gpt-5",
            "reasoning": {"effort": "high"},
            "input": "hi",
        }));
        assert_eq!(p.translated.reasoning_effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn passthrough_top_p_into_extra_body() {
        let p = parse(json!({
            "model": "gpt-5",
            "input": "hi",
            "top_p": 0.9,
        }));
        assert_eq!(p.translated.extra_body.get("top_p"), Some(&json!(0.9)));
    }
}
