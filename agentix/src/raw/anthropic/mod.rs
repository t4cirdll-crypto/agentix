pub mod request;
pub mod response;

use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use tracing::debug;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::{PostConfig, post_json, post_streaming};
use crate::raw::shared::ToolDefinition;
use crate::request::{Message, ToolCall};
use crate::types::{
    CompleteResponse, FinishReason, PartialToolCall, StreamBufs, ToolCallChunk, UsageStats,
};

use response::{ContentBlockDelta, ContentBlockStart, ResponseBlock, StreamEvent};

pub(crate) async fn stream_anthropic(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    stream_anthropic_with_post_config(
        token,
        http,
        config,
        messages,
        tools,
        PostConfig {
            use_query_key: false,
            auth_header: Some("x-api-key"),
            extra_headers: &[("anthropic-version", "2023-06-01")],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await
}

pub(crate) async fn stream_mimo(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    stream_anthropic_with_post_config(
        token,
        http,
        config,
        messages,
        tools,
        PostConfig {
            use_query_key: false,
            auth_header: Some("api-key"),
            extra_headers: &[],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await
}

async fn stream_anthropic_with_post_config(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    post_config: PostConfig,
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    let req = request::build_anthropic_request(config, messages, tools, true);
    let url = format!("{}/v1/messages", config.base_url.trim_end_matches('/'));
    let resp = post_streaming(http, &url, &req, token, &post_config).await?;

    Ok(async_stream::stream! {
        let mut bufs = StreamBufs::new();
        let mut blocks: Vec<Option<BlockBuild>> = Vec::new();
        let mut sse  = resp.bytes_stream().eventsource();
        let mut saw_message_stop = false;

        while let Some(ev_res) = sse.next().await {
            match ev_res {
                Ok(ev) => {
                    #[cfg(feature = "sensitive-logs")]
                    if crate::sensitive_logs_enabled() {
                        tracing::info!(body = %ev.data, "received raw streaming response chunk");
                    }
                    if ev.data == "[DONE]" {
                        break;
                    }
                    match serde_json::from_str::<StreamEvent>(&ev.data) {
                        Ok(chunk) => {
                            if matches!(chunk, StreamEvent::MessageStop) {
                                saw_message_stop = true;
                            }
                            for lev in parse_stream_event(chunk, &mut bufs, &mut blocks) { yield lev; }
                        }
                        Err(e) => {
                            debug!(data = %ev.data, error = %e, "anthropic chunk parse failed");
                        }
                    }
                }
                Err(e) => { yield LlmEvent::Error(e.to_string()); break; }
            }
        }
        if !saw_message_stop {
            yield LlmEvent::Error("stream ended without message_stop".to_string());
        }
        for tc in finalize(&mut bufs) { yield LlmEvent::ToolCall(tc); }
        // Only emit provider_data when the turn has both thinking AND tool_use
        // blocks — that's the combination where Anthropic enforces signature
        // round-trip on the next turn.
        if let Some(state) = assistant_state_from_blocks(&blocks) {
            yield LlmEvent::AssistantState(state);
        }
        yield LlmEvent::Done;
    }
    .boxed())
}

pub(crate) async fn complete_anthropic(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<CompleteResponse, ApiError> {
    complete_anthropic_with_post_config(
        token,
        http,
        config,
        messages,
        tools,
        PostConfig {
            use_query_key: false,
            auth_header: Some("x-api-key"),
            extra_headers: &[("anthropic-version", "2023-06-01")],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await
}

pub(crate) async fn complete_mimo(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<CompleteResponse, ApiError> {
    complete_anthropic_with_post_config(
        token,
        http,
        config,
        messages,
        tools,
        PostConfig {
            use_query_key: false,
            auth_header: Some("api-key"),
            extra_headers: &[],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await
}

async fn complete_anthropic_with_post_config(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    post_config: PostConfig,
) -> Result<CompleteResponse, ApiError> {
    let req = request::build_anthropic_request(config, messages, tools, false);
    let url = format!("{}/v1/messages", config.base_url.trim_end_matches('/'));
    let body = post_json(http, &url, &req, token, &post_config).await?;

    // Parse twice: once structurally for content/tool_calls/reasoning, once
    // as a raw Value to preserve the full content array for round-tripping
    // thinking blocks with signatures.
    let raw_value: serde_json::Value = serde_json::from_str(&body).map_err(ApiError::Json)?;
    let raw: response::Response = serde_json::from_str(&body).map_err(ApiError::Json)?;

    let mut content_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut tool_calls = Vec::new();
    let mut has_thinking = false;

    for block in &raw.content {
        match block {
            ResponseBlock::Text { text } => content_buf.push_str(text),
            ResponseBlock::Thinking { thinking, .. } => {
                reasoning_buf.push_str(thinking);
                has_thinking = true;
            }
            ResponseBlock::RedactedThinking { .. } => {
                has_thinking = true;
            }
            ResponseBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_default(),
                });
            }
        }
    }

    // Only preserve provider_data when the turn has both thinking AND tool
    // calls — that's when Anthropic enforces signature round-trip. Pure
    // thinking-no-tools turns don't require it (the API strips them).
    let provider_data = if has_thinking && !tool_calls.is_empty() {
        raw_value
            .get("content")
            .cloned()
            .map(anthropic_content_to_provider_data)
    } else {
        None
    };

    Ok(CompleteResponse {
        content: if content_buf.is_empty() {
            None
        } else {
            Some(content_buf)
        },
        reasoning: if reasoning_buf.is_empty() {
            None
        } else {
            Some(reasoning_buf)
        },
        tool_calls,
        provider_data,
        usage: raw.usage.map(UsageStats::from).unwrap_or_default(),
        finish_reason: raw
            .stop_reason
            .as_deref()
            .map(FinishReason::from)
            .unwrap_or_default(),
    })
}

/// Wrap the raw content array in a tagged envelope so the request serializer
/// can identify Anthropic's format if another provider ever emits
/// provider_data of a different shape. The wrapper also isolates callers from
/// the raw Anthropic content shape (they should treat it as opaque).
fn anthropic_content_to_provider_data(content: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "anthropic_content": content,
    })
}

fn parse_stream_event(
    ev: StreamEvent,
    bufs: &mut StreamBufs,
    blocks: &mut Vec<Option<BlockBuild>>,
) -> Vec<LlmEvent> {
    match ev {
        StreamEvent::MessageStart { message } => {
            if let Some(u) = message.usage {
                return vec![LlmEvent::Usage(UsageStats::from(u))];
            }
            vec![]
        }
        StreamEvent::MessageDelta { usage, .. } => {
            if let Some(u) = usage {
                return vec![LlmEvent::Usage(UsageStats::from(u))];
            }
            vec![]
        }
        StreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            let idx = index as usize;
            if bufs.tool_call_bufs.len() <= idx {
                bufs.tool_call_bufs.resize_with(idx + 1, || None);
            }
            if blocks.len() <= idx {
                blocks.resize_with(idx + 1, || None);
            }
            match content_block {
                ContentBlockStart::ToolUse { id, name } => {
                    bufs.tool_call_bufs[idx] = Some(PartialToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: String::new(),
                    });
                    blocks[idx] = Some(BlockBuild::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input_json: String::new(),
                    });
                    vec![LlmEvent::ToolCallChunk(ToolCallChunk {
                        id,
                        name,
                        delta: String::new(),
                        index,
                    })]
                }
                ContentBlockStart::Text { text } => {
                    blocks[idx] = Some(BlockBuild::Text { text: text.clone() });
                    if text.is_empty() {
                        vec![]
                    } else {
                        bufs.content_buf.push_str(&text);
                        vec![LlmEvent::Token(text)]
                    }
                }
                ContentBlockStart::Thinking { thinking } => {
                    blocks[idx] = Some(BlockBuild::Thinking {
                        text: thinking.clone(),
                        signature: None,
                    });
                    if thinking.is_empty() {
                        vec![]
                    } else {
                        bufs.reasoning_buf.push_str(&thinking);
                        vec![LlmEvent::Reasoning(thinking)]
                    }
                }
                ContentBlockStart::RedactedThinking { data } => {
                    blocks[idx] = Some(BlockBuild::RedactedThinking { data });
                    vec![]
                }
            }
        }
        StreamEvent::ContentBlockDelta { index, delta } => {
            let idx = index as usize;
            match delta {
                ContentBlockDelta::TextDelta { text } if !text.is_empty() => {
                    if let Some(Some(BlockBuild::Text { text: t })) = blocks.get_mut(idx) {
                        t.push_str(&text);
                    }
                    bufs.content_buf.push_str(&text);
                    vec![LlmEvent::Token(text)]
                }
                ContentBlockDelta::ThinkingDelta { thinking } if !thinking.is_empty() => {
                    if let Some(Some(BlockBuild::Thinking { text, .. })) = blocks.get_mut(idx) {
                        text.push_str(&thinking);
                    }
                    bufs.reasoning_buf.push_str(&thinking);
                    vec![LlmEvent::Reasoning(thinking)]
                }
                ContentBlockDelta::SignatureDelta { signature } => {
                    if let Some(Some(BlockBuild::Thinking { signature: sig, .. })) =
                        blocks.get_mut(idx)
                    {
                        match sig {
                            Some(existing) => existing.push_str(&signature),
                            None => *sig = Some(signature),
                        }
                    }
                    vec![]
                }
                ContentBlockDelta::InputJsonDelta { partial_json } if !partial_json.is_empty() => {
                    if let Some(Some(BlockBuild::ToolUse { input_json, .. })) = blocks.get_mut(idx)
                    {
                        input_json.push_str(&partial_json);
                    }
                    if let Some(Some(partial)) = bufs.tool_call_bufs.get_mut(idx) {
                        partial.arguments.push_str(&partial_json);
                        vec![LlmEvent::ToolCallChunk(ToolCallChunk {
                            id: partial.id.clone(),
                            name: partial.name.clone(),
                            delta: partial_json,
                            index,
                        })]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            }
        }
        _ => vec![],
    }
}

#[derive(Debug)]
enum BlockBuild {
    Text {
        text: String,
    },
    Thinking {
        text: String,
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
}

impl BlockBuild {
    fn to_value(&self) -> serde_json::Value {
        match self {
            BlockBuild::Text { text } => {
                serde_json::json!({ "type": "text", "text": text })
            }
            BlockBuild::Thinking { text, signature } => {
                let mut obj = serde_json::json!({ "type": "thinking", "thinking": text });
                if let Some(sig) = signature {
                    obj.as_object_mut()
                        .unwrap()
                        .insert("signature".into(), serde_json::Value::String(sig.clone()));
                }
                obj
            }
            BlockBuild::RedactedThinking { data } => {
                serde_json::json!({ "type": "redacted_thinking", "data": data })
            }
            BlockBuild::ToolUse {
                id,
                name,
                input_json,
            } => {
                let input: serde_json::Value = serde_json::from_str(input_json)
                    .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                serde_json::json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input,
                })
            }
        }
    }
}

fn assistant_state_from_blocks(blocks: &[Option<BlockBuild>]) -> Option<serde_json::Value> {
    let has_thinking = blocks.iter().flatten().any(|b| {
        matches!(
            b,
            BlockBuild::Thinking { .. } | BlockBuild::RedactedThinking { .. }
        )
    });
    let has_tool_use = blocks
        .iter()
        .flatten()
        .any(|b| matches!(b, BlockBuild::ToolUse { .. }));
    if !(has_thinking && has_tool_use) {
        return None;
    }
    let arr: Vec<serde_json::Value> = blocks.iter().flatten().map(|b| b.to_value()).collect();
    Some(serde_json::json!({ "anthropic_content": serde_json::Value::Array(arr) }))
}

fn finalize(bufs: &mut StreamBufs) -> Vec<ToolCall> {
    bufs.tool_call_bufs
        .drain(..)
        .flatten()
        .map(|p| ToolCall {
            id: p.id,
            name: p.name,
            arguments: p.arguments,
        })
        .collect()
}
