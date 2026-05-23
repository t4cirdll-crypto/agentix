//! Translate an inbound Chat Completions wire request into agentix's
//! internal `Request` + `Vec<Message>` representation.

use serde_json::Value;

use crate::raw::shared::{FunctionDefinition, ToolDefinition, ToolKind};
use crate::request::{
    Content, ImageContent, ImageData, Message, ReasoningEffort, ToolCall, ToolChoice, UserContent,
};
use crate::server::translated::Translated;

use super::error::OpenAIError;
use super::wire;

/// Forward-passthrough fields that map to agentix `extra_body` for providers
/// that support them. We keep the original keys verbatim.
const PASSTHROUGH_KEYS: &[&str] = &[
    "top_p",
    "top_k",
    "stop",
    "n",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
];

pub fn translate(incoming: wire::ChatCompletionsRequest) -> Result<Translated, OpenAIError> {
    let stream = incoming.stream.unwrap_or(false);
    let mut system_prompt: Option<String> = None;
    let mut messages: Vec<Message> = Vec::with_capacity(incoming.messages.len());

    for wm in incoming.messages {
        match wm {
            wire::RequestMessage::System { content, .. } => {
                let text = flatten_text(content);
                if !text.is_empty() {
                    // Concatenate consecutive system messages with a blank line.
                    match &mut system_prompt {
                        Some(prev) => {
                            prev.push_str("\n\n");
                            prev.push_str(&text);
                        }
                        None => system_prompt = Some(text),
                    }
                }
            }
            wire::RequestMessage::User { content, .. } => {
                let parts = parts_from_content(content)?;
                if !parts.is_empty() {
                    messages.push(Message::User(parts));
                }
            }
            wire::RequestMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
                ..
            } => {
                let text = content.map(flatten_text).filter(|s| !s.is_empty());
                let reasoning = reasoning_content.filter(|s| !s.is_empty());
                let tool_calls: Vec<ToolCall> = tool_calls
                    .unwrap_or_default()
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    })
                    .collect();
                messages.push(Message::Assistant {
                    content: text,
                    reasoning,
                    tool_calls,
                    provider_data: None,
                });
            }
            wire::RequestMessage::Tool {
                content,
                tool_call_id,
            } => {
                let parts = parts_from_content(content)?;
                let result_content = if parts.is_empty() {
                    vec![Content::text("")]
                } else {
                    parts
                        .into_iter()
                        .map(|uc| match uc {
                            UserContent::Text { text } => Content::text(text),
                            UserContent::Image(img) => Content::Image(img),
                            UserContent::Document(_) => Content::text(""),
                        })
                        .collect()
                };
                messages.push(Message::ToolResult {
                    call_id: tool_call_id,
                    content: result_content,
                });
            }
        }
    }

    let tools = incoming
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|t| ToolDefinition {
            kind: ToolKind::Function,
            function: FunctionDefinition {
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters,
                strict: t.function.strict,
            },
        })
        .collect();

    let tool_choice = incoming
        .tool_choice
        .map(translate_tool_choice)
        .transpose()?;

    let max_tokens = incoming
        .max_completion_tokens
        .or(incoming.max_tokens)
        .unwrap_or(4096);

    let reasoning_effort = incoming
        .reasoning_effort
        .as_deref()
        .and_then(parse_reasoning_effort);

    let mut extra_body = serde_json::Map::new();
    for (k, v) in incoming.extra {
        if PASSTHROUGH_KEYS.contains(&k.as_str()) {
            extra_body.insert(k, v);
        }
    }

    Ok(Translated {
        system_prompt,
        model_from_client: incoming.model,
        max_tokens,
        messages,
        tools,
        tool_choice,
        temperature: incoming.temperature,
        reasoning_effort,
        stream,
        extra_body,
    })
}

fn flatten_text(c: wire::TextOrParts) -> String {
    match c {
        wire::TextOrParts::Text(s) => s,
        wire::TextOrParts::Parts(parts) => parts
            .into_iter()
            .filter_map(|p| match p {
                wire::ContentPart::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

fn parts_from_content(c: wire::TextOrParts) -> Result<Vec<UserContent>, OpenAIError> {
    let parts = match c {
        wire::TextOrParts::Text(s) => {
            return Ok(if s.is_empty() {
                Vec::new()
            } else {
                vec![UserContent::Text { text: s }]
            });
        }
        wire::TextOrParts::Parts(p) => p,
    };

    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        match p {
            wire::ContentPart::Text { text } => {
                if !text.is_empty() {
                    out.push(UserContent::Text { text });
                }
            }
            wire::ContentPart::ImageUrl { image_url } => {
                let (data, mime_type) = parse_image_url(&image_url.url);
                out.push(UserContent::Image(ImageContent { data, mime_type }));
            }
            wire::ContentPart::Other => {
                // Unknown / unsupported part — drop with no error so we
                // tolerate provider-specific extensions.
            }
        }
    }
    Ok(out)
}

/// Parse an image URL. `data:image/png;base64,XXX` → Base64 + mime;
/// otherwise a public URL.
fn parse_image_url(url: &str) -> (ImageData, String) {
    if let Some(rest) = url.strip_prefix("data:")
        && let Some((meta, data)) = rest.split_once(',')
    {
        let mime = meta.split(';').next().unwrap_or("").to_string();
        return (ImageData::Base64(data.to_string()), mime);
    }
    (ImageData::Url(url.to_string()), String::new())
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
        wire::ToolChoice::Tool { function, .. } => ToolChoice::Tool(function.name),
    })
}

fn parse_reasoning_effort(s: &str) -> Option<ReasoningEffort> {
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

// suppress warning: extra is read via flatten then iterated
#[allow(dead_code)]
fn _silence_value_use(_: &Value) {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(body: serde_json::Value) -> Translated {
        let req: wire::ChatCompletionsRequest = serde_json::from_value(body).unwrap();
        translate(req).unwrap()
    }

    #[test]
    fn system_message_becomes_system_prompt() {
        let t = parse(json!({
            "model": "x",
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ]
        }));
        assert_eq!(t.system_prompt.as_deref(), Some("be helpful"));
        assert_eq!(t.messages.len(), 1);
    }

    #[test]
    fn multiple_system_messages_concatenated() {
        let t = parse(json!({
            "model": "x",
            "messages": [
                {"role": "system", "content": "rule a"},
                {"role": "system", "content": "rule b"},
                {"role": "user", "content": "hi"},
            ]
        }));
        assert_eq!(t.system_prompt.as_deref(), Some("rule a\n\nrule b"));
    }

    #[test]
    fn assistant_with_reasoning_and_tool_calls() {
        let t = parse(json!({
            "model": "x",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "let me think",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{\"a\":1}"}
                    }]
                }
            ]
        }));
        let asst = t
            .messages
            .iter()
            .find_map(|m| match m {
                Message::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                    ..
                } => Some((content.clone(), reasoning.clone(), tool_calls.clone())),
                _ => None,
            })
            .unwrap();
        assert_eq!(asst.0, None);
        assert_eq!(asst.1.as_deref(), Some("let me think"));
        assert_eq!(asst.2.len(), 1);
        assert_eq!(asst.2[0].id, "call_1");
        assert_eq!(asst.2[0].name, "calc");
    }

    #[test]
    fn tool_message_becomes_tool_result() {
        let t = parse(json!({
            "model": "x",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "calc", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "42"}
            ]
        }));
        let last = t.messages.last().unwrap();
        match last {
            Message::ToolResult { call_id, content } => {
                assert_eq!(call_id, "call_1");
                let text = match &content[0] {
                    Content::Text { text } => text.as_str(),
                    _ => "",
                };
                assert_eq!(text, "42");
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn user_content_parts_with_image_url() {
        let t = parse(json!({
            "model": "x",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
                ]
            }]
        }));
        assert_eq!(t.messages.len(), 1);
        let parts = match &t.messages[0] {
            Message::User(p) => p.clone(),
            _ => panic!(),
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[1], UserContent::Image(img) if img.mime_type == "image/png"));
    }

    #[test]
    fn reasoning_effort_high_maps() {
        let t = parse(json!({
            "model":"x",
            "reasoning_effort":"high",
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.reasoning_effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn passthrough_top_p_seed_into_extra_body() {
        let t = parse(json!({
            "model":"x",
            "top_p": 0.8,
            "seed": 42,
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.extra_body.get("top_p"), Some(&json!(0.8)));
        assert_eq!(t.extra_body.get("seed"), Some(&json!(42)));
    }

    #[test]
    fn tool_choice_named_required() {
        let t = parse(json!({
            "model":"x",
            "tool_choice":"required",
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert!(matches!(t.tool_choice, Some(ToolChoice::Required)));
    }

    #[test]
    fn tool_choice_named_function() {
        let t = parse(json!({
            "model":"x",
            "tool_choice":{"type":"function","function":{"name":"calc"}},
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert!(matches!(t.tool_choice, Some(ToolChoice::Tool(ref n)) if n == "calc"));
    }

    #[test]
    fn max_completion_tokens_preferred_when_present() {
        let t = parse(json!({
            "model":"x",
            "max_tokens": 100,
            "max_completion_tokens": 500,
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.max_tokens, 500);
    }
}
