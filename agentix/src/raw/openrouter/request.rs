use serde::Serialize;
use serde_json::Value;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{DocumentData, ImageData, Message, ReasoningEffort, ToolChoice, UserContent};

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<crate::raw::shared::ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<crate::request::ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<crate::raw::shared::ResponseFormat>,
    /// OpenRouter's unified reasoning control — normalizes across underlying
    /// providers. See <https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningParam>,
    #[serde(flatten)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct ReasoningParam {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<&'static str>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum OaiMessage {
    System {
        content: MessageContent,
    },
    User {
        content: MessageContent,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ResponseToolCall>,
        /// Opaque typed entries from a previous turn — round-tripped verbatim
        /// so the underlying provider's reasoning/signature state survives.
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_details: Option<Vec<Value>>,
    },
    Tool {
        tool_call_id: String,
        content: ToolMessageContent,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ToolMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ImageUrl {
        image_url: ImageUrl,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// OpenRouter `file` content part (PDF plugin). `file.file_data` is a
    /// data URL (`data:application/pdf;base64,...`) or a public URL.
    /// `file.filename` is required by some underlying providers.
    File {
        file: FilePart,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Debug, Serialize, Clone)]
pub struct FilePart {
    pub filename: String,
    pub file_data: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct CacheControl {
    pub r#type: String,
}

impl CacheControl {
    fn ephemeral() -> Self {
        CacheControl {
            r#type: "ephemeral".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ResponseFunctionCall,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseFunctionCall {
    pub name: String,
    pub arguments: String,
}

pub(crate) fn build_openrouter_request(
    config: &AgentConfig,
    history: Vec<Message>,
    tools: &[ToolDefinition],
    tool_choice: Option<ToolChoice>,
    stream: bool,
) -> Request {
    let mut messages = Vec::new();
    if let Some(s) = &config.system_prompt
        && !s.is_empty()
    {
        messages.push(OaiMessage::System {
            content: MessageContent::Text(s.clone()),
        });
    }
    for m in history {
        match m {
            Message::User(parts) => messages.push(OaiMessage::User {
                content: user_content_from_parts(parts),
            }),
            Message::Assistant {
                content,
                tool_calls,
                provider_data,
                ..
            } => {
                let reasoning_details = provider_data
                    .as_ref()
                    .and_then(|v| v.get("openrouter_reasoning_details"))
                    .and_then(|v| v.as_array())
                    .cloned();
                messages.push(OaiMessage::Assistant {
                    content,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tc| ResponseToolCall {
                            id: tc.id,
                            r#type: "function".to_string(),
                            function: ResponseFunctionCall {
                                name: tc.name,
                                arguments: tc.arguments,
                            },
                        })
                        .collect(),
                    reasoning_details,
                });
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                let tool_content = if let [Content::Text { text }] = content.as_slice() {
                    ToolMessageContent::Text(text.clone())
                } else {
                    // Tool-result documents are not accepted by the OpenAI-
                    // chat-compatible proxy shape OpenRouter exposes, so we
                    // drop them — documents belong in user messages where the
                    // `file` part is supported.
                    let parts = content
                        .iter()
                        .filter_map(|p| match p {
                            Content::Text { text } => Some(ContentPart::Text {
                                text: text.clone(),
                                cache_control: None,
                            }),
                            Content::Image(img) => {
                                let url = match &img.data {
                                    ImageData::Base64(b) => {
                                        format!("data:{};base64,{}", img.mime_type, b)
                                    }
                                    ImageData::Url(u) => u.clone(),
                                };
                                Some(ContentPart::ImageUrl {
                                    image_url: ImageUrl { url, detail: None },
                                    cache_control: None,
                                })
                            }
                            Content::Document(_) => None,
                        })
                        .collect();
                    ToolMessageContent::Parts(parts)
                };
                messages.push(OaiMessage::Tool {
                    tool_call_id: call_id.clone(),
                    content: tool_content,
                });
            }
        }
    }

    stamp_cache_breakpoints(&mut messages);

    let tools_opt = if tools.is_empty() {
        None
    } else {
        Some(tools.to_vec())
    };
    let extra = if config.extra_body.is_empty() {
        None
    } else {
        Some(config.extra_body.clone())
    };
    Request {
        model: config.model.clone(),
        messages,
        tools: tools_opt,
        tool_choice,
        stream: Some(stream),
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        response_format: config
            .response_format
            .clone()
            .map(crate::raw::shared::ResponseFormat::from),
        reasoning: openrouter_reasoning(config.reasoning_effort),
        extra_body: extra,
    }
}

/// Map the cross-provider effort enum to OpenRouter's unified
/// `reasoning.effort`. OpenRouter normalizes across underlying providers, so
/// we can pass through any of the standard levels. `None` omits the block
/// entirely (underlying provider's default takes over).
fn openrouter_reasoning(effort: Option<ReasoningEffort>) -> Option<ReasoningParam> {
    let e = effort?;
    let wire = match e {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
        ReasoningEffort::Max => "max",
    };
    Some(ReasoningParam { effort: Some(wire) })
}

fn user_content_from_parts(parts: Vec<UserContent>) -> MessageContent {
    let blocks: Vec<ContentPart> = parts
        .into_iter()
        .map(|p| match p {
            UserContent::Text { text: t } => ContentPart::Text {
                text: t,
                cache_control: None,
            },
            UserContent::Image(img) => {
                let url = match img.data {
                    ImageData::Url(u) => u,
                    ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
                };
                ContentPart::ImageUrl {
                    image_url: ImageUrl { url, detail: None },
                    cache_control: None,
                }
            }
            UserContent::Document(doc) => {
                let file_data = match doc.data {
                    DocumentData::Url(u) => u,
                    DocumentData::Base64(b) => format!("data:{};base64,{}", doc.mime_type, b),
                };
                ContentPart::File {
                    file: FilePart {
                        filename: doc
                            .filename
                            .unwrap_or_else(|| default_filename(&doc.mime_type)),
                        file_data,
                    },
                    cache_control: None,
                }
            }
        })
        .collect();

    if let [ContentPart::Text { text, .. }] = blocks.as_slice() {
        return MessageContent::Text(text.clone());
    }
    MessageContent::Parts(blocks)
}

fn default_filename(mime: &str) -> String {
    match mime {
        "application/pdf" => "document.pdf".into(),
        _ => "document".into(),
    }
}

// Stamp cache_control: ephemeral on system prompt, first user message (summary), and last user message (latest turn).
fn stamp_cache_breakpoints(messages: &mut [OaiMessage]) {
    let mut first_user: Option<usize> = None;
    let mut last_user: Option<usize> = None;

    for (i, msg) in messages.iter_mut().enumerate() {
        match msg {
            OaiMessage::System { content } => stamp_cache(content),
            OaiMessage::User { .. } => {
                first_user.get_or_insert(i);
                last_user = Some(i);
            }
            _ => {}
        }
    }

    if let Some(f) = first_user {
        if let OaiMessage::User { content } = &mut messages[f] {
            stamp_cache(content);
        }
        if let Some(l) = last_user.filter(|&l| l != f)
            && let OaiMessage::User { content } = &mut messages[l]
        {
            stamp_cache(content);
        }
    }
}

fn stamp_cache(content: &mut MessageContent) {
    match content {
        MessageContent::Text(text) => {
            *content = MessageContent::Parts(vec![ContentPart::Text {
                text: text.clone(),
                cache_control: Some(CacheControl::ephemeral()),
            }]);
        }
        MessageContent::Parts(parts) => {
            if let Some(last) = parts.last_mut() {
                set_cache_control(last);
            }
        }
    }
}

fn set_cache_control(part: &mut ContentPart) {
    match part {
        ContentPart::Text { cache_control, .. }
        | ContentPart::ImageUrl { cache_control, .. }
        | ContentPart::File { cache_control, .. } => {
            *cache_control = Some(CacheControl::ephemeral());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Content as MsgContent, Message, ToolCall};

    fn cfg() -> AgentConfig {
        AgentConfig {
            model: "anthropic/claude-sonnet-4.6".into(),
            ..Default::default()
        }
    }

    #[test]
    fn reasoning_effort_maps_to_unified_param() {
        let mut c = cfg();
        c.reasoning_effort = Some(ReasoningEffort::High);
        let req = build_openrouter_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["reasoning"]["effort"], "high");
    }

    #[test]
    fn reasoning_effort_none_maps_to_none_string() {
        let mut c = cfg();
        c.reasoning_effort = Some(ReasoningEffort::None);
        let req = build_openrouter_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["reasoning"]["effort"], "none");
    }

    #[test]
    fn provider_data_attaches_reasoning_details_to_assistant() {
        let pd = serde_json::json!({
            "openrouter_reasoning_details": [
                {"type": "reasoning.encrypted", "data": "ENC_A", "format": "anthropic-claude-v1"},
                {"type": "reasoning.text", "text": "step 1", "signature": "SIG_A"}
            ]
        });
        let history = vec![
            Message::User(vec![MsgContent::text("hi")]),
            Message::Assistant {
                content: Some("...".into()),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "x".into(),
                    arguments: "{}".into(),
                }],
                provider_data: Some(pd),
            },
        ];
        let req = build_openrouter_request(&cfg(), history, &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();
        let assistant = messages.iter().find(|m| m["role"] == "assistant").unwrap();
        let rd = assistant["reasoning_details"].as_array().unwrap();
        assert_eq!(rd.len(), 2);
        assert_eq!(rd[0]["type"], "reasoning.encrypted");
        assert_eq!(rd[0]["data"], "ENC_A");
        assert_eq!(rd[1]["signature"], "SIG_A");
    }

    #[test]
    fn document_base64_emits_file_part() {
        use crate::request::{DocumentContent, DocumentData, UserContent};
        let history = vec![Message::User(vec![
            UserContent::Text {
                text: "read".into(),
            },
            UserContent::Document(DocumentContent {
                data: DocumentData::Base64("UERGZmFrZQ==".into()),
                mime_type: "application/pdf".into(),
                filename: Some("spec.pdf".into()),
            }),
        ])];
        let req = build_openrouter_request(&cfg(), history, &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        let user = json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "user")
            .unwrap();
        let parts = user["content"].as_array().unwrap();
        let file_part = parts.iter().find(|p| p["type"] == "file").unwrap();
        assert_eq!(file_part["file"]["filename"], "spec.pdf");
        assert_eq!(
            file_part["file"]["file_data"],
            "data:application/pdf;base64,UERGZmFrZQ=="
        );
    }
}
