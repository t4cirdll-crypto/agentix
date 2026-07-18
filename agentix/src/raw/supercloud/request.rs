//! Request types for the SuperCloud (Render proxy / NDJSON) API.
//!
//! The SuperCloud proxy accepts a JSON object with `messages`, `provider`,
//! `model`, and optional `tools`. Tools are serialised as an *object*
//! (not an array) keyed by tool name, unlike OpenAI/DeepSeek which use
//! arrays.

use serde::Serialize;
use serde_json::Value;

use crate::raw::shared::ToolDefinition;
use crate::request::{Content, ImageData, Message};

// ── Wire types ────────────────────────────────────────────────────────────────

/// Top-level request body sent to `POST /api/ai/chat`.
#[derive(Debug, Serialize)]
pub(crate) struct Request {
    pub messages: Vec<MessageWire>,
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// A single message in the conversation, matching the OpenAI-compatible
/// format that the SuperCloud proxy expects.
#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub(crate) enum MessageWire {
    #[serde(rename = "system")]
    System {
        content: String,
    },
    #[serde(rename = "user")]
    User {
        /// Content as a plain string for single text, or JSON array for
        /// multi-part messages. Uses `serde_json::Value` so serde picks
        /// the right wire format automatically.
        content: Value,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCallWire>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolCallWire {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCallWire,
}

#[derive(Debug, Serialize)]
pub(crate) struct FunctionCallWire {
    pub name: String,
    pub arguments: String,
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Build a SuperCloud request from agentix `Message`s and `ToolDefinition`s.
pub(crate) fn build_supercloud_request(
    model: &str,
    provider: &str,
    system_prompt: Option<&str>,
    history: Vec<Message>,
    tools: &[ToolDefinition],
    stream: bool,
) -> Request {
    let mut messages = Vec::new();

    // System prompt first
    if let Some(s) = system_prompt.filter(|s| !s.is_empty()) {
        messages.push(MessageWire::System {
            content: s.to_string(),
        });
    }

    // Conversation history
    for m in history {
        match m {
            Message::User(parts) => {
                let content: Value = if parts.len() == 1 {
                    if let Content::Text { text } = &parts[0] {
                        // Single text part → plain string (not array)
                        Value::String(text.clone())
                    } else {
                        // Single image → array with one element
                        let arr: Vec<Value> = parts.into_iter().filter_map(|p| content_part_to_value(p)).collect();
                        Value::Array(arr)
                    }
                } else {
                    // Multiple parts → array
                    let arr: Vec<Value> = parts.into_iter().filter_map(|p| content_part_to_value(p)).collect();
                    Value::Array(arr)
                };
                messages.push(MessageWire::User { content });
            }
            Message::Assistant {
                content,
                reasoning,
                tool_calls,
                ..
            } => {
                messages.push(MessageWire::Assistant {
                    content,
                    reasoning,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tc| ToolCallWire {
                            id: tc.id,
                            kind: "function".to_string(),
                            function: FunctionCallWire {
                                name: tc.name,
                                arguments: tc.arguments,
                            },
                        })
                        .collect(),
                });
            }
            Message::ToolResult { call_id, content } => {
                // Concatenate text parts into a single string for tool results.
                let text: String = content
                    .iter()
                    .filter_map(|p| match p {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                messages.push(MessageWire::Tool {
                    content: text,
                    tool_call_id: call_id,
                });
            }
        }
    }

    // Convert tools from the agentix array format to the SuperCloud
    // object format (keyed by tool name).
    let tools_value = if tools.is_empty() {
        None
    } else {
        let mut map = serde_json::Map::new();
        for tool in tools {
            let tool_obj = serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                }
            });
            map.insert(tool.function.name.clone(), tool_obj);
        }
        Some(Value::Object(map))
    };

    Request {
        messages,
        provider: provider.to_string(),
        model: model.to_string(),
        tools: tools_value,
        stream: if stream { Some(true) } else { None },
    }
}

/// Convert a single agentix `Content` part into a JSON value for the
/// OpenAI-compatible content array format.
fn content_part_to_value(part: Content) -> Option<Value> {
    match part {
        Content::Text { text } => Some(Value::String(text)),
        Content::Image(img) => {
            let url = match img.data {
                ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
                ImageData::Url(u) => u,
            };
            Some(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": url }
            }))
        }
        Content::Document(_) => None,
    }
}
