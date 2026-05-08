//! Translate an inbound Anthropic Messages wire request into agentix's
//! internal `Request` + `Vec<Message>` representation.

use serde_json::Value;

use crate::raw::anthropic::request as wire;
use crate::raw::shared::{FunctionDefinition, ToolDefinition, ToolKind};
use crate::request::{
    Content, DocumentContent, DocumentData, ImageContent, ImageData, Message, ReasoningEffort,
    ToolCall, ToolChoice, UserContent,
};

use super::error::{ErrorKind, ServerError};

/// Anthropic system field. Accepts either a plain string or a list of system
/// blocks. The agentix `Request::system_message` is a single string; we flatten
/// blocks by concatenating their text and dropping `cache_control` (the
/// outbound Anthropic adapter re-stamps its own cache breakpoints, and
/// non-Anthropic backends ignore cache control entirely).
#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum SystemField {
    Text(String),
    Blocks(Vec<wire::SystemBlock>),
}

impl SystemField {
    fn flatten(self) -> Option<String> {
        let s = match self {
            SystemField::Text(s) => s,
            SystemField::Blocks(blocks) => blocks
                .into_iter()
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n\n"),
        };
        if s.is_empty() { None } else { Some(s) }
    }
}

/// What we accept on the wire. Mirrors a subset of Anthropic's Messages
/// request schema; unknown fields are tolerated and ignored.
#[derive(Debug, serde::Deserialize)]
pub struct IncomingRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<wire::RequestMessage>,
    #[serde(default)]
    pub system: Option<SystemField>,
    #[serde(default)]
    pub tools: Option<Vec<wire::Tool>>,
    #[serde(default)]
    pub tool_choice: Option<wire::ToolChoice>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub thinking: Option<wire::ThinkingConfig>,
    #[serde(default)]
    pub output_config: Option<wire::OutputConfig>,
    /// `top_p`, `top_k`, `stop_sequences`, `metadata`, `service_tier` are
    /// captured here so they can be forwarded into agentix `extra_body` for
    /// providers that accept top-level passthrough.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// Translation result.
pub struct Translated {
    pub system_prompt: Option<String>,
    pub model_from_client: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
    pub temperature: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub stream: bool,
    pub extra_body: serde_json::Map<String, Value>,
}

/// Forward-passthrough fields from Anthropic's request body that map to
/// `extra_body` in agentix's Request. Different providers honour different
/// subsets; we trust the upstream serializer to pick what it understands.
const PASSTHROUGH_KEYS: &[&str] = &[
    "top_p",
    "top_k",
    "stop_sequences",
    "metadata",
    "service_tier",
];

pub fn translate(incoming: IncomingRequest) -> Result<Translated, ServerError> {
    let stream = incoming.stream.unwrap_or(false);
    let system_prompt = incoming.system.and_then(|s| s.flatten());

    let mut messages: Vec<Message> = Vec::with_capacity(incoming.messages.len());
    for wm in incoming.messages {
        push_translated_message(&mut messages, wm)?;
    }

    let tools = incoming
        .tools
        .unwrap_or_default()
        .into_iter()
        .map(|t| ToolDefinition {
            kind: ToolKind::Function,
            function: FunctionDefinition {
                name: t.name,
                description: t.description,
                parameters: t.input_schema,
                strict: None,
            },
        })
        .collect();

    let tool_choice = incoming.tool_choice.map(translate_tool_choice);
    let reasoning_effort = translate_reasoning(incoming.thinking, incoming.output_config);

    let mut extra_body = serde_json::Map::new();
    for (k, v) in incoming.extra {
        if PASSTHROUGH_KEYS.contains(&k.as_str()) {
            extra_body.insert(k, v);
        }
    }

    Ok(Translated {
        system_prompt,
        model_from_client: incoming.model,
        max_tokens: incoming.max_tokens,
        messages,
        tools,
        tool_choice,
        temperature: incoming.temperature,
        reasoning_effort,
        stream,
        extra_body,
    })
}

fn push_translated_message(
    out: &mut Vec<Message>,
    wm: wire::RequestMessage,
) -> Result<(), ServerError> {
    let blocks: Vec<wire::ContentBlock> = match wm.content {
        wire::MessageContent::Text(t) => vec![wire::ContentBlock::Text {
            text: t,
            cache_control: None,
        }],
        wire::MessageContent::Blocks(b) => b,
    };

    match wm.role.as_str() {
        "user" => translate_user_blocks(out, blocks),
        "assistant" => {
            translate_assistant_blocks(out, blocks);
            Ok(())
        }
        other => Err(ServerError::new(
            ErrorKind::InvalidRequest,
            format!("unexpected message role: {other}"),
        )),
    }
}

/// Anthropic places `tool_result` blocks INSIDE user-role messages, possibly
/// mixed with text/image/document blocks. agentix represents tool results as
/// separate `Message::ToolResult` entries. Lift each tool_result block out;
/// remaining non-tool blocks form a single `Message::User`. Order is preserved.
fn translate_user_blocks(
    out: &mut Vec<Message>,
    blocks: Vec<wire::ContentBlock>,
) -> Result<(), ServerError> {
    let mut user_parts: Vec<UserContent> = Vec::new();
    for block in blocks {
        match block {
            wire::ContentBlock::Text { text, .. } => {
                if !user_parts.is_empty() || !text.is_empty() {
                    user_parts.push(UserContent::Text { text });
                }
            }
            wire::ContentBlock::Image { source, .. } => {
                let (data, mime_type) = match source {
                    wire::ImageSource::Base64 { media_type, data } => {
                        (ImageData::Base64(data), media_type)
                    }
                    wire::ImageSource::Url { url } => (ImageData::Url(url), String::new()),
                };
                user_parts.push(UserContent::Image(ImageContent { data, mime_type }));
            }
            wire::ContentBlock::Document { source, .. } => {
                let (data, mime_type) = match source {
                    wire::DocumentSource::Base64 { media_type, data } => {
                        (DocumentData::Base64(data), media_type)
                    }
                    wire::DocumentSource::Url { url } => (DocumentData::Url(url), String::new()),
                };
                user_parts.push(UserContent::Document(DocumentContent {
                    data,
                    mime_type,
                    filename: None,
                }));
            }
            wire::ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                // Flush any accumulated user-side parts into their own message
                // first so we preserve the source order.
                if !user_parts.is_empty() {
                    out.push(Message::User(std::mem::take(&mut user_parts)));
                }
                let result_content = match content {
                    wire::ToolResultContent::Text(t) => vec![Content::text(t)],
                    wire::ToolResultContent::Parts(parts) => parts
                        .into_iter()
                        .map(|p| match p {
                            wire::ToolResultPart::Text { text } => Content::text(text),
                            wire::ToolResultPart::Image { source } => {
                                let (data, mime_type) = match source {
                                    wire::ImageSource::Base64 { media_type, data } => {
                                        (ImageData::Base64(data), media_type)
                                    }
                                    wire::ImageSource::Url { url } => {
                                        (ImageData::Url(url), String::new())
                                    }
                                };
                                Content::Image(ImageContent { data, mime_type })
                            }
                        })
                        .collect(),
                };
                out.push(Message::ToolResult {
                    call_id: tool_use_id,
                    content: result_content,
                });
            }
            wire::ContentBlock::Thinking { .. }
            | wire::ContentBlock::RedactedThinking { .. }
            | wire::ContentBlock::ToolUse { .. } => {
                // Anthropic's spec doesn't put these on user messages; tolerate
                // and drop rather than error.
            }
        }
    }
    if !user_parts.is_empty() {
        out.push(Message::User(user_parts));
    }
    Ok(())
}

/// Translate an assistant message's blocks into a `Message::Assistant`.
/// When any thinking block is present we ALSO stash the full block array into
/// `provider_data["anthropic_content"]` so signatures round-trip verbatim back
/// to Anthropic-compatible upstreams. (Non-Anthropic upstreams ignore
/// `provider_data`, which is the correct silent-drop behaviour.)
fn translate_assistant_blocks(out: &mut Vec<Message>, blocks: Vec<wire::ContentBlock>) {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut has_thinking = false;

    for block in &blocks {
        match block {
            wire::ContentBlock::Text { text, .. } => content.push_str(text),
            wire::ContentBlock::Thinking { thinking, .. } => {
                reasoning.push_str(thinking);
                has_thinking = true;
            }
            wire::ContentBlock::RedactedThinking { .. } => {
                has_thinking = true;
            }
            wire::ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_default(),
                });
            }
            // Tool results don't belong here; ignore.
            wire::ContentBlock::ToolResult { .. }
            | wire::ContentBlock::Image { .. }
            | wire::ContentBlock::Document { .. } => {}
        }
    }

    let provider_data = if has_thinking {
        let arr: Vec<Value> = blocks
            .iter()
            .map(|b| serde_json::to_value(b).unwrap_or(Value::Null))
            .collect();
        Some(serde_json::json!({ "anthropic_content": arr }))
    } else {
        None
    };

    out.push(Message::Assistant {
        content: if content.is_empty() {
            None
        } else {
            Some(content)
        },
        reasoning: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        tool_calls,
        provider_data,
    });
}

fn translate_tool_choice(tc: wire::ToolChoice) -> ToolChoice {
    match tc {
        wire::ToolChoice::Auto => ToolChoice::Auto,
        wire::ToolChoice::Any => ToolChoice::Required,
        wire::ToolChoice::Tool { name } => ToolChoice::Tool(name),
    }
}

/// Anthropic's `thinking` (Adaptive/Disabled) plus `output_config.effort`
/// collapse to a single `ReasoningEffort` value.
fn translate_reasoning(
    thinking: Option<wire::ThinkingConfig>,
    output_config: Option<wire::OutputConfig>,
) -> Option<ReasoningEffort> {
    match (thinking, output_config) {
        (Some(wire::ThinkingConfig::Disabled), _) => Some(ReasoningEffort::None),
        (Some(wire::ThinkingConfig::Adaptive), Some(cfg)) => {
            Some(match cfg.effort {
                wire::AnthropicEffort::Low => ReasoningEffort::Low,
                wire::AnthropicEffort::Medium => ReasoningEffort::Medium,
                wire::AnthropicEffort::High => ReasoningEffort::High,
                wire::AnthropicEffort::XHigh => ReasoningEffort::XHigh,
                wire::AnthropicEffort::Max => ReasoningEffort::Max,
            })
        }
        (Some(wire::ThinkingConfig::Adaptive), None) => Some(ReasoningEffort::Medium),
        (None, Some(cfg)) => Some(match cfg.effort {
            wire::AnthropicEffort::Low => ReasoningEffort::Low,
            wire::AnthropicEffort::Medium => ReasoningEffort::Medium,
            wire::AnthropicEffort::High => ReasoningEffort::High,
            wire::AnthropicEffort::XHigh => ReasoningEffort::XHigh,
            wire::AnthropicEffort::Max => ReasoningEffort::Max,
        }),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(body: serde_json::Value) -> Translated {
        let incoming: IncomingRequest = serde_json::from_value(body).unwrap();
        translate(incoming).unwrap()
    }

    #[test]
    fn flatten_string_system() {
        let t = parse(json!({
            "model":"x","max_tokens":1,
            "system":"be helpful",
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.system_prompt.as_deref(), Some("be helpful"));
    }

    #[test]
    fn flatten_block_system() {
        let t = parse(json!({
            "model":"x","max_tokens":1,
            "system":[{"type":"text","text":"a"},{"type":"text","text":"b"}],
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.system_prompt.as_deref(), Some("a\n\nb"));
    }

    #[test]
    fn tool_result_split_out_of_user_message() {
        let t = parse(json!({
            "model":"x","max_tokens":1,
            "messages":[
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"tu1","content":"42"},
                    {"type":"tool_result","tool_use_id":"tu2","content":"43"},
                    {"type":"text","text":"and now please continue"}
                ]}
            ]
        }));
        assert_eq!(t.messages.len(), 3);
        assert!(matches!(&t.messages[0], Message::ToolResult { call_id, .. } if call_id == "tu1"));
        assert!(matches!(&t.messages[1], Message::ToolResult { call_id, .. } if call_id == "tu2"));
        assert!(matches!(&t.messages[2], Message::User(_)));
    }

    #[test]
    fn assistant_thinking_preserved_in_provider_data() {
        let t = parse(json!({
            "model":"x","max_tokens":1,
            "messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[
                    {"type":"thinking","thinking":"plan...","signature":"sig-A"},
                    {"type":"tool_use","id":"tu1","name":"calc","input":{"a":1}}
                ]}
            ]
        }));
        assert_eq!(t.messages.len(), 2);
        if let Message::Assistant { provider_data, tool_calls, reasoning, .. } = &t.messages[1] {
            assert_eq!(reasoning.as_deref(), Some("plan..."));
            assert_eq!(tool_calls.len(), 1);
            let pd = provider_data.as_ref().expect("provider_data must be set");
            let arr = pd.get("anthropic_content").and_then(|v| v.as_array()).unwrap();
            assert_eq!(arr.len(), 2);
            assert_eq!(arr[0]["signature"], "sig-A");
        } else {
            panic!("expected assistant message");
        }
    }

    #[test]
    fn thinking_disabled_maps_to_reasoning_none() {
        let t = parse(json!({
            "model":"x","max_tokens":1,
            "thinking":{"type":"disabled"},
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.reasoning_effort, Some(ReasoningEffort::None));
    }

    #[test]
    fn output_effort_high_maps_to_high() {
        let t = parse(json!({
            "model":"x","max_tokens":1,
            "thinking":{"type":"adaptive"},
            "output_config":{"effort":"high"},
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.reasoning_effort, Some(ReasoningEffort::High));
    }

    #[test]
    fn passthrough_top_p_into_extra_body() {
        let t = parse(json!({
            "model":"x","max_tokens":1,
            "top_p":0.9,
            "stop_sequences":["END"],
            "messages":[{"role":"user","content":"hi"}]
        }));
        assert_eq!(t.extra_body.get("top_p"), Some(&json!(0.9)));
        assert_eq!(t.extra_body.get("stop_sequences"), Some(&json!(["END"])));
    }
}
