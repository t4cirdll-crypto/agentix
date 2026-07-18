//! Layer-2 stream transform for the SuperCloud (Render proxy / NDJSON) API.
//!
//! Consumes a raw NDJSON byte stream and produces [`LlmEvent`]s.
//! The proxy speaks newline-delimited JSON (one JSON object per line),
//! NOT SSE (Server-Sent Events), so parsing is simpler than for
//! OpenAI/Anthropic providers.
//!
//! # NDJSON Format
//!
//! ```text
//! {"type":"text","content":"Hello"}
//! {"type":"reasoning","content":"thinking..."}
//! {"type":"tool-call","toolName":"get_weather","args":{"city":"London"},"toolCallId":"call_xxx"}
//! {"type":"finish","reason":"stop","usage":{...}}
//! ```

use futures::{StreamExt, stream::BoxStream};
use tracing::debug;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::provider::{PostConfig, post_json, post_streaming};
use crate::raw::shared::ToolDefinition;
use crate::request::{Message, ToolCall};
use crate::types::{CompleteResponse, FinishReason, UsageStats};

use request::build_supercloud_request;
use response::NdjsonEvent;

mod request;
mod response;

// ── Public API ────────────────────────────────────────────────────────────────

/// Send a streaming request to the SuperCloud proxy and return an
/// [`LlmEvent`] stream.
///
/// The `api_key` parameter is the OAuth bearer token obtained from
/// `~/.better-auth/token.json`.
pub(crate) async fn stream_supercloud(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    let req = build_supercloud_request(
        &config.model,
        "concentrateai",
        config.system_prompt.as_deref(),
        messages.to_vec(),
        tools,
        true,
    );

    let base_url = config.base_url.trim_end_matches('/');
    let url = format!("{}/api/ai/chat", base_url);

    let resp = post_streaming(
        http,
        &url,
        &req,
        token,
        &PostConfig {
            use_query_key: false,
            auth_header: None,
            extra_headers: &[("Content-Type", "application/json")],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await?;

    Ok(parse_ndjson_stream(resp, config.model.clone()))
}

/// Send a non-streaming request to the SuperCloud proxy.
pub(crate) async fn complete_supercloud(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<CompleteResponse, ApiError> {
    let req = build_supercloud_request(
        &config.model,
        "concentrateai",
        config.system_prompt.as_deref(),
        messages.to_vec(),
        tools,
        false,
    );

    let base_url = config.base_url.trim_end_matches('/');
    let url = format!("{}/api/ai/chat", base_url);

    let body = post_json(
        http,
        &url,
        &req,
        token,
        &PostConfig {
            use_query_key: false,
            auth_header: None,
            extra_headers: &[("Content-Type", "application/json")],
            max_retries: config.max_retries,
            retry_delay_ms: config.retry_delay_ms,
        },
    )
    .await?;

    // For non-streaming, the SuperCloud proxy may still return NDJSON
    // (it does not support a true non-streaming mode). Parse it the
    // same way as streaming but collect into a CompleteResponse.
    let events = parse_ndjson_body(&body);
    collect_complete(events, &config.model)
}

// ── NDJSON Parser ─────────────────────────────────────────────────────────────

/// Parse a `reqwest::Response` body as an NDJSON stream and produce
/// [`LlmEvent`]s.
fn parse_ndjson_stream(
    resp: reqwest::Response,
    _model: String,
) -> BoxStream<'static, LlmEvent> {
    async_stream::stream! {
        let mut buf = String::new();
        let mut stream = resp.bytes_stream();

        // Accumulators for tool calls (SuperCloud emits complete tool calls,
        // not deltas, so we collect them at finish).
        let mut content_acc = String::new();
        let mut reasoning_acc = String::new();
        let mut tool_calls_acc: Vec<(String, String, String)> = Vec::new(); // (id, name, args_json)
        let mut final_usage: Option<UsageStats> = None;

        while let Some(chunk_res) = stream.next().await {
            match chunk_res {
                Ok(bytes) => {
                    // Safety: proxy returns UTF-8.
                    let chunk_str = String::from_utf8_lossy(&bytes);
                    buf.push_str(&chunk_str);

                    // Process complete lines
                    loop {
                        match buf.find('\n') {
                            Some(newline_pos) => {
                                let line = buf[..newline_pos].trim().to_string();
                                buf = buf[newline_pos + 1..].to_string();

                                if line.is_empty() {
                                    continue;
                                }

                                match serde_json::from_str::<NdjsonEvent>(&line) {
                                    Ok(event) => {
                                        match event {
                                            NdjsonEvent::Text { content } => {
                                                content_acc.push_str(&content);
                                                yield LlmEvent::Token(content);
                                            }
                                            NdjsonEvent::Reasoning { content } => {
                                                reasoning_acc.push_str(&content);
                                                yield LlmEvent::Reasoning(content);
                                            }
                                            NdjsonEvent::ToolCall { tool_name, args, tool_call_id } => {
                                                let args_str = serde_json::to_string(&args)
                                                    .unwrap_or_default();
                                                tool_calls_acc.push((tool_call_id.clone(), tool_name.clone(), args_str.clone()));
                                                // Emit a complete tool call event
                                                yield LlmEvent::ToolCall(ToolCall {
                                                    id: tool_call_id,
                                                    name: tool_name,
                                                    arguments: args_str,
                                                });
                                            }
                                            NdjsonEvent::Finish { reason, usage, .. } => {
                                                if let Some(u) = usage {
                                                    let stats = UsageStats {
                                                        prompt_tokens: u.input_tokens.unwrap_or(0),
                                                        completion_tokens: u.output_tokens.unwrap_or(0),
                                                        total_tokens: u.total_tokens.unwrap_or(0),
                                                        cache_read_tokens: u.input_token_details
                                                            .as_ref()
                                                            .and_then(|d| d.cache_read_tokens)
                                                            .unwrap_or(0),
                                                        cache_creation_tokens: u.input_token_details
                                                            .as_ref()
                                                            .and_then(|d| d.cache_write_tokens)
                                                            .unwrap_or(0),
                                                        reasoning_tokens: u.output_token_details
                                                            .as_ref()
                                                            .and_then(|d| d.reasoning_tokens)
                                                            .unwrap_or(0),
                                                    };
                                                    final_usage = Some(stats.clone());
                                                    yield LlmEvent::Usage(stats);
                                                }
                                                // Map finish reason
                                                let _finish_reason = match reason.as_str() {
                                                    "stop" | "end_turn" => FinishReason::Stop,
                                                    "length" | "max_tokens" => FinishReason::Length,
                                                    "tool_calls" | "tool_use" => FinishReason::ToolCalls,
                                                    _ => FinishReason::Other(reason),
                                                };
                                                // Note: we already emitted individual ToolCall events
                                                // during parsing, so no need to re-emit here.
                                                yield LlmEvent::Done;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        debug!(line = %line, error = %e, "supercloud ndjson parse failed, skipping");
                                    }
                                }
                            }
                            None => break, // wait for more data
                        }
                    }
                }
                Err(e) => {
                    yield LlmEvent::Error(format!("supercloud stream error: {e}"));
                    break;
                }
            }
        }

        // If the stream ended without a finish event, emit a synthetic Done.
        // (The proxy should always send a finish event, but be defensive.)
        if final_usage.is_none() {
            yield LlmEvent::Done;
        }
    }
    .boxed()
}

/// Parse a complete NDJSON body (non-streaming fallback).
fn parse_ndjson_body(body: &str) -> Vec<NdjsonEvent> {
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<NdjsonEvent>(l.trim()).ok())
        .collect()
}

/// Collect NDJSON events into a [`CompleteResponse`].
fn collect_complete(events: Vec<NdjsonEvent>, _model: &str) -> Result<CompleteResponse, ApiError> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let mut usage = UsageStats::default();
    let mut finish_reason = FinishReason::Stop;

    for event in events {
        match event {
            NdjsonEvent::Text { content: c } => content.push_str(&c),
            NdjsonEvent::Reasoning { content: r } => reasoning.push_str(&r),
            NdjsonEvent::ToolCall { tool_name, args, tool_call_id } => {
                let args_str = serde_json::to_string(&args).unwrap_or_default();
                tool_calls.push(ToolCall {
                    id: tool_call_id,
                    name: tool_name,
                    arguments: args_str,
                });
            }
            NdjsonEvent::Finish { reason, usage: u, .. } => {
                if let Some(u) = u {
                    usage = UsageStats {
                        prompt_tokens: u.input_tokens.unwrap_or(0),
                        completion_tokens: u.output_tokens.unwrap_or(0),
                        total_tokens: u.total_tokens.unwrap_or(0),
                        cache_read_tokens: u.input_token_details
                            .as_ref()
                            .and_then(|d| d.cache_read_tokens)
                            .unwrap_or(0),
                        cache_creation_tokens: u.input_token_details
                            .as_ref()
                            .and_then(|d| d.cache_write_tokens)
                            .unwrap_or(0),
                        reasoning_tokens: u.output_token_details
                            .as_ref()
                            .and_then(|d| d.reasoning_tokens)
                            .unwrap_or(0),
                    };
                }
                finish_reason = match reason.as_str() {
                    "stop" | "end_turn" => FinishReason::Stop,
                    "length" | "max_tokens" => FinishReason::Length,
                    "tool_calls" | "tool_use" => FinishReason::ToolCalls,
                    _ => FinishReason::Other(reason),
                };
            }
        }
    }

    Ok(CompleteResponse {
        content: if content.is_empty() { None } else { Some(content) },
        reasoning: if reasoning.is_empty() { None } else { Some(reasoning) },
        tool_calls,
        provider_data: None,
        usage,
        finish_reason,
    })
}
