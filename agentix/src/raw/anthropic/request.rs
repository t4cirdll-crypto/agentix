use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{DocumentData, ImageData, Message, ReasoningEffort, UserContent};

// ── Cache control ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String,
}

impl CacheControl {
    fn ephemeral() -> Self {
        CacheControl {
            kind: "ephemeral".to_string(),
        }
    }
}

// ── System prompt (block format required for cache_control) ───────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cache_control: Option<CacheControl>,
}

// ── Request ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Request {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_config: Option<OutputConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    Adaptive,
    Disabled,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputConfig {
    pub effort: AnthropicEffort,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RequestMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// Content payload for a `tool_result` block.
/// Anthropic accepts either a plain string or an array of text/image parts.
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

/// Merge consecutive `Message::User` entries into one by concatenating their
/// content parts. Anthropic requires strict user/assistant alternation.
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

pub(crate) fn build_anthropic_request(
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
                        role: "user".into(),
                        content: MessageContent::Blocks(std::mem::take(&mut pending_tool_results)),
                    });
                }
                out_messages.push(RequestMessage {
                    role: "user".into(),
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
                        role: "user".into(),
                        content: MessageContent::Blocks(std::mem::take(&mut pending_tool_results)),
                    });
                }
                // If we have provider_data from a previous Anthropic turn with
                // thinking+tool_use, it's the authoritative wire form — emit
                // it verbatim to preserve thinking-block ordering relative to
                // tool_use blocks (required by Anthropic's signature check).
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
                        role: "assistant".into(),
                        content: MessageContent::Blocks(parsed),
                    });
                } else if tool_calls.is_empty() {
                    out_messages.push(RequestMessage {
                        role: "assistant".into(),
                        content: MessageContent::Text(content.clone().unwrap_or_default()),
                    });
                } else {
                    let mut blocks: Vec<ContentBlock> = Vec::new();
                    if let Some(t) = content
                        && !t.is_empty()
                    {
                        blocks.push(ContentBlock::Text {
                            text: t.clone(),
                            cache_control: None,
                        });
                    }
                    for tc in tool_calls {
                        let input = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                        blocks.push(ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input,
                            cache_control: None,
                        });
                    }
                    out_messages.push(RequestMessage {
                        role: "assistant".into(),
                        content: MessageContent::Blocks(blocks),
                    });
                }
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                let wire_content = if let [Content::Text { text }] = content.as_slice() {
                    ToolResultContent::Text(text.clone())
                } else {
                    // Anthropic `tool_result.content` only accepts text + image
                    // parts. Documents in tool results are dropped; documents
                    // belong on user messages where the `document` block is
                    // supported.
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
                    cache_control: None,
                });
            }
        }
    }
    if !pending_tool_results.is_empty() {
        out_messages.push(RequestMessage {
            role: "user".into(),
            content: MessageContent::Blocks(pending_tool_results),
        });
    }

    stamp_cache_breakpoints(&mut out_messages);

    let anthropic_tools: Option<Vec<Tool>> = if tools.is_empty() {
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

    // System prompt as blocks so we can attach cache_control.
    let system = config
        .system_prompt
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| {
            vec![SystemBlock {
                kind: "text".to_string(),
                text: s.to_string(),
                cache_control: Some(CacheControl::ephemeral()),
            }]
        });

    let (thinking, output_config) = anthropic_thinking(config.reasoning_effort);

    Request {
        model: config.model.clone(),
        max_tokens: config.max_tokens.unwrap_or(32_768),
        messages: out_messages,
        system,
        tools: anthropic_tools,
        tool_choice,
        stream: Some(stream),
        temperature: config.temperature,
        thinking,
        output_config,
    }
}

/// Translate the cross-provider `ReasoningEffort` into Anthropic's
/// `thinking` + `output_config.effort` pair. `None` (unset) emits neither
/// field, leaving the model's default in place.
fn anthropic_thinking(
    effort: Option<ReasoningEffort>,
) -> (Option<ThinkingConfig>, Option<OutputConfig>) {
    match effort {
        None => (None, None),
        Some(ReasoningEffort::None) => (Some(ThinkingConfig::Disabled), None),
        Some(e) => {
            let anth = match e {
                ReasoningEffort::None => unreachable!(),
                // Anthropic has no `minimal`; coerce down to `low`.
                ReasoningEffort::Minimal | ReasoningEffort::Low => AnthropicEffort::Low,
                ReasoningEffort::Medium => AnthropicEffort::Medium,
                ReasoningEffort::High => AnthropicEffort::High,
                ReasoningEffort::XHigh => AnthropicEffort::XHigh,
                ReasoningEffort::Max => AnthropicEffort::Max,
            };
            (
                Some(ThinkingConfig::Adaptive),
                Some(OutputConfig { effort: anth }),
            )
        }
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
            UserContent::Text { text: t } => ContentBlock::Text {
                text: t,
                cache_control: None,
            },
            UserContent::Image(img) => ContentBlock::Image {
                source: match img.data {
                    ImageData::Base64(data) => ImageSource::Base64 {
                        media_type: img.mime_type,
                        data,
                    },
                    ImageData::Url(url) => ImageSource::Url { url },
                },
                cache_control: None,
            },
            UserContent::Document(doc) => ContentBlock::Document {
                source: match doc.data {
                    DocumentData::Base64(data) => DocumentSource::Base64 {
                        media_type: doc.mime_type,
                        data,
                    },
                    DocumentData::Url(url) => DocumentSource::Url { url },
                },
                cache_control: None,
            },
        })
        .collect();
    if has_non_text && !has_text {
        blocks.push(ContentBlock::Text {
            text: " ".to_string(),
            cache_control: None,
        });
    }
    MessageContent::Blocks(blocks)
}

// ── Cache breakpoints ─────────────────────────────────────────────────────────
//
// Strategy (mirrors OpenRouter):
//   • first user message  → breakpoint (covers compact summary / stable history head)
//   • last  user message  → breakpoint (covers current turn; warms cache for next request)
//
// System prompt already gets cache_control in build_anthropic_request above.

fn stamp_cache_breakpoints(messages: &mut [RequestMessage]) {
    let mut first_user: Option<usize> = None;
    let mut last_user: Option<usize> = None;

    for (i, msg) in messages.iter().enumerate() {
        if msg.role == "user" {
            first_user.get_or_insert(i);
            last_user = Some(i);
        }
    }

    if let Some(f) = first_user {
        stamp_cache(&mut messages[f].content);
        if let Some(l) = last_user.filter(|&l| l != f) {
            stamp_cache(&mut messages[l].content);
        }
    }
}

fn stamp_cache(content: &mut MessageContent) {
    match content {
        MessageContent::Text(text) => {
            *content = MessageContent::Blocks(vec![ContentBlock::Text {
                text: text.clone(),
                cache_control: Some(CacheControl::ephemeral()),
            }]);
        }
        MessageContent::Blocks(blocks) => {
            if let Some(block) = blocks.last_mut() {
                set_cache_control(block);
            }
        }
    }
}

fn set_cache_control(block: &mut ContentBlock) {
    match block {
        ContentBlock::Text { cache_control, .. }
        | ContentBlock::Image { cache_control, .. }
        | ContentBlock::Document { cache_control, .. }
        | ContentBlock::ToolUse { cache_control, .. }
        | ContentBlock::ToolResult { cache_control, .. } => {
            *cache_control = Some(CacheControl::ephemeral());
        }
        // Thinking blocks don't take cache_control; skip silently.
        ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Content, Message};

    fn cfg(system: &str) -> AgentConfig {
        AgentConfig {
            system_prompt: Some(system.into()),
            model: "claude-haiku-4-5-20251001".into(),
            ..Default::default()
        }
    }

    #[test]
    fn system_block_has_cache_control() {
        let req = build_anthropic_request(&cfg("Be helpful."), &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let blocks = json["system"].as_array().expect("system must be array");
        let last = blocks.last().unwrap();
        assert_eq!(last["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn single_user_message_gets_breakpoint() {
        let msgs = vec![Message::User(vec![Content::text("Hello")])];
        let req = build_anthropic_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let user = json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "user")
            .unwrap()
            .clone();
        // After stamping, content becomes an array.
        let blocks = user["content"]
            .as_array()
            .expect("must be blocks after stamp");
        let text_block = blocks.iter().find(|b| b["type"] == "text").unwrap();
        assert_eq!(text_block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn multi_turn_first_and_last_stamped() {
        let msgs = vec![
            Message::User(vec![Content::text("First")]),
            Message::Assistant {
                content: Some("A".into()),
                reasoning: None,
                tool_calls: vec![],
                provider_data: None,
            },
            Message::User(vec![Content::text("Second")]),
        ];
        let req = build_anthropic_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();

        let users: Vec<_> = messages.iter().filter(|m| m["role"] == "user").collect();
        assert_eq!(users.len(), 2);

        for u in &users {
            let blocks = u["content"].as_array().expect("must be blocks");
            let text = blocks.iter().find(|b| b["type"] == "text").unwrap();
            assert_eq!(
                text["cache_control"]["type"], "ephemeral",
                "both user messages must be stamped"
            );
        }
    }

    #[test]
    fn no_system_prompt_no_system_field() {
        let config = AgentConfig {
            model: "m".into(),
            ..Default::default()
        };
        let req = build_anthropic_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(
            json["system"].is_null(),
            "absent system prompt must not serialize"
        );
    }

    #[test]
    fn reasoning_effort_none_emits_thinking_disabled() {
        let mut config = cfg("S");
        config.reasoning_effort = Some(ReasoningEffort::None);
        let req = build_anthropic_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["thinking"]["type"], "disabled");
        assert!(json["output_config"].is_null());
    }

    #[test]
    fn reasoning_effort_high_emits_adaptive_with_effort() {
        let mut config = cfg("S");
        config.reasoning_effort = Some(ReasoningEffort::High);
        let req = build_anthropic_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["thinking"]["type"], "adaptive");
        assert_eq!(json["output_config"]["effort"], "high");
    }

    #[test]
    fn reasoning_effort_minimal_collapses_to_low() {
        let mut config = cfg("S");
        config.reasoning_effort = Some(ReasoningEffort::Minimal);
        let req = build_anthropic_request(&config, &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["output_config"]["effort"], "low");
    }

    #[test]
    fn no_reasoning_effort_omits_thinking_field() {
        let req = build_anthropic_request(&cfg("S"), &[], &[], false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["thinking"].is_null());
        assert!(json["output_config"].is_null());
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
        let req = build_anthropic_request(&cfg("S"), &msgs, &[], false);
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

    #[test]
    fn document_url_emits_url_source() {
        use crate::request::{DocumentContent, DocumentData, UserContent};
        let msgs = vec![Message::User(vec![
            UserContent::Text {
                text: "summarize".into(),
            },
            UserContent::Document(DocumentContent {
                data: DocumentData::Url("https://example.com/doc.pdf".into()),
                mime_type: "application/pdf".into(),
                filename: None,
            }),
        ])];
        let req = build_anthropic_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let user = json["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "user")
            .unwrap();
        let blocks = user["content"].as_array().unwrap();
        let doc = blocks.iter().find(|b| b["type"] == "document").unwrap();
        assert_eq!(doc["source"]["type"], "url");
        assert_eq!(doc["source"]["url"], "https://example.com/doc.pdf");
    }

    #[test]
    fn provider_data_anthropic_content_is_emitted_verbatim() {
        // Simulate a previous assistant turn with interleaved thinking +
        // tool_use. On round-trip, Anthropic requires this exact ordering.
        let pd = serde_json::json!({
            "anthropic_content": [
                {"type": "thinking", "thinking": "step 1", "signature": "sig-A"},
                {"type": "tool_use", "id": "tu_1", "name": "x", "input": {"q": "a"}},
                {"type": "thinking", "thinking": "step 2", "signature": "sig-B"},
                {"type": "tool_use", "id": "tu_2", "name": "x", "input": {"q": "b"}},
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
        let req = build_anthropic_request(&cfg("S"), &msgs, &[], false);
        let json = serde_json::to_value(&req).unwrap();
        let messages = json["messages"].as_array().unwrap();
        let assistant = messages.iter().find(|m| m["role"] == "assistant").unwrap();
        let blocks = assistant["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["signature"], "sig-A");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "tu_1");
        assert_eq!(blocks[2]["type"], "thinking");
        assert_eq!(blocks[2]["signature"], "sig-B");
        assert_eq!(blocks[3]["type"], "tool_use");
        assert_eq!(blocks[3]["id"], "tu_2");
    }
}
