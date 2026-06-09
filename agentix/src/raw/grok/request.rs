use serde::Serialize;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{ImageData, Message, ToolChoice, UserContent};

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,
    pub messages: Vec<GrokMessage>,
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
    #[serde(flatten)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum GrokMessage {
    System {
        content: String,
    },
    User {
        content: UserMessageContent,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ResponseToolCall>,
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
pub enum UserMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
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

pub(crate) fn build_grok_request(
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
        messages.push(GrokMessage::System { content: s.clone() });
    }
    for m in history {
        match m {
            Message::User(parts) => messages.push(GrokMessage::User {
                content: user_content_from_parts(parts),
            }),
            Message::Assistant {
                content,
                reasoning,
                tool_calls,
                ..
            } => {
                messages.push(GrokMessage::Assistant {
                    content,
                    reasoning_content: reasoning,
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
                });
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                let tool_content = if let [Content::Text { text }] = content.as_slice() {
                    ToolMessageContent::Text(text.clone())
                } else {
                    // Grok chat-completions doesn't accept file parts —
                    // drop Document entries silently.
                    let parts = content
                        .iter()
                        .filter_map(|p| match p {
                            Content::Text { text } => {
                                Some(ContentPart::Text { text: text.clone() })
                            }
                            Content::Image(img) => {
                                let url = match &img.data {
                                    ImageData::Base64(b) => {
                                        format!("data:{};base64,{}", img.mime_type, b)
                                    }
                                    ImageData::Url(u) => u.clone(),
                                };
                                Some(ContentPart::ImageUrl {
                                    image_url: ImageUrl { url, detail: None },
                                })
                            }
                            Content::Document(_) => None,
                        })
                        .collect();
                    ToolMessageContent::Parts(parts)
                };
                messages.push(GrokMessage::Tool {
                    tool_call_id: call_id,
                    content: tool_content,
                });
            }
        }
    }
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
        extra_body: extra,
    }
}

fn user_content_from_parts(parts: Vec<UserContent>) -> UserMessageContent {
    let blocks: Vec<ContentPart> = parts
        .into_iter()
        .filter_map(|p| match p {
            UserContent::Text { text: t } => Some(ContentPart::Text { text: t }),
            UserContent::Image(img) => {
                let url = match img.data {
                    ImageData::Url(u) => u,
                    ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
                };
                Some(ContentPart::ImageUrl {
                    image_url: ImageUrl { url, detail: None },
                })
            }
            // Grok has no wire for files — silently drop.
            UserContent::Document(_) => None,
        })
        .collect();

    if let [ContentPart::Text { text }] = blocks.as_slice() {
        return UserMessageContent::Text(text.clone());
    }
    UserMessageContent::Parts(blocks)
}
