use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{DocumentData, ImageData, Message, ReasoningEffort, UserContent};

// ── Request ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,
    /// Mimo says max_tokens is optional with per-model defaults
    /// (v2.5-pro/v2-pro = 131072, v2-flash = 65536, others = 32768). Omit
    /// when the user hasn't set one — the server picks the right ceiling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub messages: Vec<RequestMessage>,
    /// Mimo accepts `system` as either a plain string or an array of blocks;
    /// we send the string form (Mimo doesn't document `cache_control` on
    /// system blocks, so the array form has no advantage here).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// Mimo's thinking config — only `enabled` / `disabled` per the docs.
/// (Anthropic-side `adaptive` is intentionally absent.)
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    Enabled,
    Disabled,
}

#[derive(Debug, Serialize)]
pub struct RequestMessage {
    pub role: &'static str,
    pub content: MessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    Document {
        source: DocumentSource,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
    },
}

/// Content payload for a `tool_result` block — string or array of parts.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Parts(Vec<ToolResultPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultPart {
    Text { text: String },
    Image { source: ImageSource },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Serialize)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Mimo's tool_choice only allows `auto`. We don't expose `Any` / `Tool` here —
/// even if the user picks them upstream, callers should map to `Auto` before
/// invoking Mimo (build_mimo_request does the simple heuristic of always Auto
/// when tools are present).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
}

/// Merge consecutive `Message::User` entries into one by concatenating their
/// content parts. The Anthropic-compatible API requires strict user/assistant
/// alternation.
fn merge_consecutive_user(messages: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    for msg in messages {
        if let Message::User(parts) = msg
            && let Some(Message::User(prev)) = out.last_mut()
        {
            prev.extend(parts.iter().cloned());
            continue;
        }
        out.push(msg.clone());
    }
    out
}

pub(crate) fn build_mimo_request(
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    stream: bool,
) -> Request {
    let messages = merge_consecutive_user(messages);
    let mut out_messages: Vec<RequestMessage> = Vec::new();
    let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

    for msg in &messages {
        match msg {
            Message::User(parts) => {
                if !pending_tool_results.is_empty() {
                    out_messages.push(RequestMessage {
                        role: "user",
                        content: MessageContent::Blocks(std::mem::take(&mut pending_tool_results)),
                    });
                }
                out_messages.push(RequestMessage {
                    role: "user",
                    content: user_content_from_parts(parts.clone()),
                });
            }
            Message::Assistant {
                content,
                tool_calls,
                provider_data,
                ..
            } => {
                if !pending_tool_results.is_empty() {
                    out_messages.push(RequestMessage {
                        role: "user",
                        content: MessageContent::Blocks(std::mem::take(&mut pending_tool_results)),
                    });
                }
                // If we have provider_data from a previous turn carrying the
                // raw content blocks (incl. thinking + signature), emit them
                // verbatim. Mimo's docs explicitly recommend keeping all
                // historical thinking blocks across turns in extended-thinking
                // multi-turn tool calls.
                if let Some(blocks) = provider_data
                    .as_ref()
                    .and_then(|v| v.get("anthropic_content"))
                    .and_then(|v| v.as_array())
                    .filter(|a| !a.is_empty())
                {
                    let parsed: Vec<ContentBlock> = blocks
                        .iter()
                        .filter_map(|b| serde_json::from_value(b.clone()).ok())
                        .collect();
                    out_messages.push(RequestMessage {
                        role: "assistant",
                        content: MessageContent::Blocks(parsed),
                    });
                } else if tool_calls.is_empty() {
                    out_messages.push(RequestMessage {
                        role: "assistant",
                        content: MessageContent::Text(content.clone().unwrap_or_default()),
                    });
                } else {
                    let mut blocks: Vec<ContentBlock> = Vec::new();
                    if let Some(t) = content
                        && !t.is_empty()
                    {
                        blocks.push(ContentBlock::Text { text: t.clone() });
                    }
                    for tc in tool_calls {
                        let input = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                        blocks.push(ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input,
                        });
                    }
                    out_messages.push(RequestMessage {
                        role: "assistant",
                        content: MessageContent::Blocks(blocks),
                    });
                }
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                let wire_content = if let [Content::Text { text }] = content.as_slice() {
                    ToolResultContent::Text(text.clone())
                } else {
                    // tool_result.content only accepts text + image parts.
                    let parts = content
                        .iter()
                        .filter_map(|p| match p {
                            Content::Text { text } => {
                                Some(ToolResultPart::Text { text: text.clone() })
                            }
                            Content::Image(img) => {
                                let source = match &img.data {
                                    ImageData::Base64(data) => ImageSource::Base64 {
                                        media_type: img.mime_type.clone(),
                                        data: data.clone(),
                                    },
                                    ImageData::Url(url) => ImageSource::Url { url: url.clone() },
                                };
                                Some(ToolResultPart::Image { source })
                            }
                            Content::Document(_) => None,
                        })
                        .collect();
                    ToolResultContent::Parts(parts)
                };
                pending_tool_results.push(ContentBlock::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: wire_content,
                });
            }
        }
    }
    if !pending_tool_results.is_empty() {
        out_messages.push(RequestMessage {
            role: "user",
            content: MessageContent::Blocks(pending_tool_results),
        });
    }

    let mimo_tools: Option<Vec<Tool>> = if tools.is_empty() {
        None
    } else {
        Some(
            tools
                .iter()
                .map(|t| Tool {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: t.function.parameters.clone(),
                })
                .collect(),
        )
    };

    let tool_choice = if tools.is_empty() {
        None
    } else {
        Some(ToolChoice::Auto)
    };

    let system = config
        .system_prompt
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    Request {
        model: config.model.clone(),
        // Optional per spec; only emit when the caller explicitly asked.
        max_tokens: config.max_tokens,
        messages: out_messages,
        system,
        tools: mimo_tools,
        tool_choice,
        stream: Some(stream),
        temperature: config.temperature,
        thinking: mimo_thinking(config.reasoning_effort),
    }
}

/// Translate the cross-provider `ReasoningEffort` into Mimo's `thinking` field.
/// Mimo only allows `enabled` / `disabled`, so any non-`None` effort collapses
/// to `enabled`. Leaving the field absent (when no effort is set) lets Mimo
/// pick the per-model default (enabled on v2.5-pro/v2.5/v2-pro/v2-omni,
/// disabled on v2-flash).
fn mimo_thinking(effort: Option<ReasoningEffort>) -> Option<ThinkingConfig> {
    match effort {
        None => None,
        Some(ReasoningEffort::None) => Some(ThinkingConfig::Disabled),
        Some(_) => Some(ThinkingConfig::Enabled),
    }
}

fn user_content_from_parts(parts: Vec<UserContent>) -> MessageContent {
    if parts.len() == 1 && matches!(&parts[0], UserContent::Text { .. }) {
        if let UserContent::Text { text: t } = parts.into_iter().next().unwrap() {
            return MessageContent::Text(t);
        }
        unreachable!()
    }
    let has_text = parts.iter().any(|p| matches!(p, UserContent::Text { .. }));
    let has_non_text = parts
        .iter()
        .any(|p| matches!(p, UserContent::Image(_) | UserContent::Document(_)));
    let mut blocks: Vec<ContentBlock> = parts
        .into_iter()
        .map(|p| match p {
            UserContent::Text { text: t } => ContentBlock::Text { text: t },
            UserContent::Image(img) => ContentBlock::Image {
                source: match img.data {
                    ImageData::Base64(data) => ImageSource::Base64 {
                        media_type: img.mime_type,
                        data,
                    },
                    ImageData::Url(url) => ImageSource::Url { url },
                },
            },
            UserContent::Document(doc) => ContentBlock::Document {
                source: match doc.data {
                    DocumentData::Base64(data) => DocumentSource::Base64 {
                        media_type: doc.mime_type,
                        data,
                    },
                    DocumentData::Url(url) => DocumentSource::Url { url },
                },
            },
        })
        .collect();
    if has_non_text && !has_text {
        blocks.push(ContentBlock::Text {
            text: " ".to_string(),
        });
    }
    MessageContent::Blocks(blocks)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Content, Message};

    fn cfg(system: &str) -> AgentConfig {
        AgentConfig {
            system_prompt: Some(system.into()),
            model: "mimo-v2.5-pro".into(),
            ..Default::default()
        }
    }

    #[test]
    fn max_tokens_omitted_by_default() {
        let req = build_mimo_request(&cfg("S"), &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["max_tokens"].is_null(), "max_tokens must be optional");
    }

    #[test]
    fn max_tokens_passes_through_when_set() {
        let mut config = cfg("S");
        config.max_tokens = Some(8192);
        let req = build_mimo_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["max_tokens"], 8192);
    }

    #[test]
    fn system_is_plain_string() {
        let req = build_mimo_request(&cfg("Be helpful."), &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["system"], "Be helpful.");
    }

    #[test]
    fn no_system_prompt_omits_field() {
        let config = AgentConfig {
            model: "mimo-v2.5-pro".into(),
            ..Default::default()
        };
        let req = build_mimo_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["system"].is_null());
    }

    #[test]
    fn no_cache_control_anywhere() {
        // Mimo's spec doesn't document cache_control on input. Make sure
        // we don't sneak one in via system, user, or tool_result.
        let config = cfg("S");
        let msgs = vec![Message::User(vec![Content::text("Hello")])];
        let req = build_mimo_request(&config, &msgs, &[], false);
        let s = serde_json::to_string(&req).unwrap();
        assert!(
            !s.contains("cache_control"),
            "no cache_control should appear: {s}"
        );
    }

    #[test]
    fn reasoning_effort_none_emits_thinking_disabled() {
        let mut config = cfg("S");
        config.reasoning_effort = Some(ReasoningEffort::None);
        let req = build_mimo_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["thinking"]["type"], "disabled");
    }

    #[test]
    fn reasoning_effort_high_emits_thinking_enabled() {
        let mut config = cfg("S");
        config.reasoning_effort = Some(ReasoningEffort::High);
        let req = build_mimo_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["thinking"]["type"], "enabled",
            "Mimo only accepts enabled/disabled, never adaptive"
        );
        assert!(
            json["output_config"].is_null(),
            "Mimo has no output_config field"
        );
    }

    #[test]
    fn no_reasoning_effort_omits_thinking_field() {
        let req = build_mimo_request(&cfg("S"), &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(
            json["thinking"].is_null(),
            "absent effort means absent thinking field — Mimo picks per-model default"
        );
    }

    #[test]
    fn provider_data_anthropic_content_round_trips_verbatim() {
        // Same envelope shape as anthropic provider, since Mimo's wire format
        // is Anthropic-compatible and signatures (if present) hash content.
        let pd = serde_json::json!({
            "anthropic_content": [
                {"type": "thinking", "thinking": "step 1", "signature": "sig-A"},
                {"type": "tool_use", "id": "tu_1", "name": "x", "input": {"q": "a"}},
            ]
        });
        let msgs = vec![
            Message::User(vec![Content::text("hi")]),
            Message::Assistant {
                content: Some("ignored".into()),
                reasoning: Some("ignored".into()),
                tool_calls: vec![crate::request::ToolCall {
                    id: "ignored".into(),
                    name: "ignored".into(),
                    arguments: "{}".into(),
                }],
                provider_data: Some(pd),
            },
        ];
        let req = build_mimo_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();
        let assistant = messages.iter().find(|m| m["role"] == "assistant").unwrap();
        let blocks = assistant["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["signature"], "sig-A");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "tu_1");
    }

    #[test]
    fn document_base64_emits_document_block() {
        use crate::request::{DocumentContent, DocumentData, UserContent};
        let msgs = vec![Message::User(vec![
            UserContent::Text {
                text: "summarize".into(),
            },
            UserContent::Document(DocumentContent {
                data: DocumentData::Base64("UERGZmFrZQ==".into()),
                mime_type: "application/pdf".into(),
                filename: None,
            }),
        ])];
        let req = build_mimo_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let user = json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "user")
            .unwrap();
        let blocks = user["content"].as_array().unwrap();
        let doc = blocks.iter().find(|b| b["type"] == "document").unwrap();
        assert_eq!(doc["source"]["type"], "base64");
        assert_eq!(doc["source"]["media_type"], "application/pdf");
        assert_eq!(doc["source"]["data"], "UERGZmFrZQ==");
    }
}
