//! Translate agentix's internal `LlmEvent` stream and `CompleteResponse`
//! values back into Anthropic Messages wire format.

use std::collections::HashSet;

use serde_json::{Value, json};

use crate::msg::LlmEvent;
use crate::raw::anthropic::response as wire;
use crate::types::{CompleteResponse, FinishReason, UsageStats};

// ── Non-streaming: CompleteResponse → Anthropic JSON Response ────────────────

/// Build the JSON body for a non-streaming `/v1/messages` response from a
/// completed agentix turn.
pub fn build_response_body(
    resp: CompleteResponse,
    request_model: &str,
    has_reasoning: bool,
) -> Value {
    let id = synth_message_id();
    let mut content_blocks: Vec<Value> = Vec::new();

    // Prefer round-tripping the upstream Anthropic block array if present;
    // it preserves signature ordering and other server-attached metadata.
    if let Some(blocks) = resp
        .provider_data
        .as_ref()
        .and_then(|v| v.get("anthropic_content"))
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
    {
        content_blocks = blocks.clone();
    } else {
        if let Some(reasoning) = resp.reasoning.as_deref().filter(|r| !r.is_empty())
            && has_reasoning
        {
            content_blocks.push(json!({
                "type": "thinking",
                "thinking": reasoning,
            }));
        }
        if let Some(text) = resp.content.as_deref().filter(|t| !t.is_empty()) {
            content_blocks.push(json!({
                "type": "text",
                "text": text,
            }));
        }
        for tc in &resp.tool_calls {
            let input: Value = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
            content_blocks.push(json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": input,
            }));
        }
    }

    let stop_reason = match resp.finish_reason {
        FinishReason::ToolCalls => "tool_use",
        FinishReason::Length => "max_tokens",
        FinishReason::ContentFilter => "stop_sequence",
        FinishReason::Stop | FinishReason::Other(_) => {
            if !resp.tool_calls.is_empty() {
                "tool_use"
            } else {
                "end_turn"
            }
        }
    };

    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": request_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": usage_to_json(&resp.usage),
    })
}

fn usage_to_json(u: &UsageStats) -> Value {
    json!({
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
        "cache_read_input_tokens": u.cache_read_tokens,
        "cache_creation_input_tokens": u.cache_creation_tokens,
    })
}

fn synth_message_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("msg_{nanos:x}")
}

// ── Streaming state machine: LlmEvent → wire SSE events ─────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse(String),
}

pub struct SseState {
    current: Option<(BlockKind, u32)>,
    /// Set after a `signature_delta` while still in a Thinking block. We hold
    /// the close until we see a non-Reasoning, non-Signature event so that
    /// `signature_delta` is always emitted BEFORE `content_block_stop` per
    /// Anthropic's wire spec.
    pending_close_thinking: bool,
    next_index: u32,
    message_started: bool,
    last_usage: Option<UsageStats>,
    has_tool_use: bool,
    open_tool_use_ids: HashSet<String>,
    model: String,
}

impl SseState {
    pub fn new(model: String) -> Self {
        Self {
            current: None,
            pending_close_thinking: false,
            next_index: 0,
            message_started: false,
            last_usage: None,
            has_tool_use: false,
            open_tool_use_ids: HashSet::new(),
            model,
        }
    }

    /// Translate one LlmEvent into zero or more wire SSE frames. Each frame
    /// is `(event_name, payload_json)`.
    pub fn on_event(&mut self, ev: LlmEvent) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        self.handle(ev, &mut out);
        out
    }

    fn ensure_message_start(&mut self, out: &mut Vec<(&'static str, Value)>) {
        if !self.message_started {
            self.message_started = true;
            out.push((
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": synth_message_id(),
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": usage_to_json(&UsageStats::default()),
                    }
                }),
            ));
        }
    }

    fn flush_pending_close(&mut self, out: &mut Vec<(&'static str, Value)>) {
        if self.pending_close_thinking
            && let Some((BlockKind::Thinking, idx)) = &self.current
        {
            out.push((
                "content_block_stop",
                json!({"type": "content_block_stop", "index": idx}),
            ));
            self.current = None;
            self.pending_close_thinking = false;
        }
    }

    fn close_current_unless(
        &mut self,
        keep: &BlockKind,
        out: &mut Vec<(&'static str, Value)>,
    ) -> bool {
        match &self.current {
            Some((kind, _)) if kind == keep => true,
            Some((_, idx)) => {
                let idx = *idx;
                out.push((
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": idx}),
                ));
                self.current = None;
                false
            }
            None => false,
        }
    }

    fn open(&mut self, kind: BlockKind, start_payload: Value, out: &mut Vec<(&'static str, Value)>) {
        let idx = self.next_index;
        self.next_index += 1;
        out.push((
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": idx,
                "content_block": start_payload,
            }),
        ));
        self.current = Some((kind, idx));
    }

    fn current_index(&self) -> Option<u32> {
        self.current.as_ref().map(|(_, i)| *i)
    }

    fn handle(&mut self, ev: LlmEvent, out: &mut Vec<(&'static str, Value)>) {
        match ev {
            LlmEvent::Token(t) => {
                self.ensure_message_start(out);
                self.flush_pending_close(out);
                if !self.close_current_unless(&BlockKind::Text, out) {
                    self.open(
                        BlockKind::Text,
                        json!({"type": "text", "text": ""}),
                        out,
                    );
                }
                let idx = self.current_index().unwrap();
                out.push((
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": t},
                    }),
                ));
            }

            LlmEvent::Reasoning(r) => {
                self.ensure_message_start(out);
                self.flush_pending_close(out);
                if !self.close_current_unless(&BlockKind::Thinking, out) {
                    self.open(
                        BlockKind::Thinking,
                        json!({"type": "thinking", "thinking": ""}),
                        out,
                    );
                }
                let idx = self.current_index().unwrap();
                out.push((
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "thinking_delta", "thinking": r},
                    }),
                ));
            }

            LlmEvent::ReasoningSignature(s) => {
                if let Some((BlockKind::Thinking, idx)) = &self.current {
                    let idx = *idx;
                    out.push((
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "signature_delta", "signature": s},
                        }),
                    ));
                    self.pending_close_thinking = true;
                }
                // If no thinking block is currently open, drop the signature.
                // It would be invalid wire output to emit signature_delta
                // outside a thinking block.
            }

            LlmEvent::ToolCallChunk(chunk) => {
                self.ensure_message_start(out);
                self.flush_pending_close(out);
                let want = BlockKind::ToolUse(chunk.id.clone());
                if !self.close_current_unless(&want, out) {
                    self.open(
                        BlockKind::ToolUse(chunk.id.clone()),
                        json!({
                            "type": "tool_use",
                            "id": chunk.id,
                            "name": chunk.name,
                            "input": {},
                        }),
                        out,
                    );
                    self.open_tool_use_ids.insert(chunk.id.clone());
                    self.has_tool_use = true;
                }
                if !chunk.delta.is_empty() {
                    let idx = self.current_index().unwrap();
                    out.push((
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": chunk.delta},
                        }),
                    ));
                }
            }

            LlmEvent::ToolCall(call) => {
                if self.open_tool_use_ids.contains(&call.id) {
                    // Already streamed via chunks; no extra wire output.
                    return;
                }
                self.ensure_message_start(out);
                self.flush_pending_close(out);
                self.close_current_unless(&BlockKind::ToolUse(call.id.clone()), out);
                let idx = self.next_index;
                self.next_index += 1;
                out.push((
                    "content_block_start",
                    json!({
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": {},
                        },
                    }),
                ));
                if !call.arguments.is_empty() {
                    out.push((
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": call.arguments},
                        }),
                    ));
                }
                out.push((
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": idx}),
                ));
                self.open_tool_use_ids.insert(call.id);
                self.has_tool_use = true;
                self.current = None;
            }

            LlmEvent::Usage(u) => {
                self.last_usage = Some(u);
            }

            LlmEvent::AssistantState(_) => {
                // Signatures travel via ReasoningSignature; AssistantState is
                // an end-of-turn redundant snapshot that we don't need on
                // the wire.
            }

            LlmEvent::Done => {
                self.ensure_message_start(out);
                self.flush_pending_close(out);
                if let Some((_, idx)) = self.current.take() {
                    out.push((
                        "content_block_stop",
                        json!({"type": "content_block_stop", "index": idx}),
                    ));
                }
                let stop_reason = if self.has_tool_use { "tool_use" } else { "end_turn" };
                let usage = self
                    .last_usage
                    .clone()
                    .unwrap_or_default();
                out.push((
                    "message_delta",
                    json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                        "usage": usage_to_json(&usage),
                    }),
                ));
                out.push((
                    "message_stop",
                    json!({"type": "message_stop"}),
                ));
            }

            LlmEvent::Error(e) => {
                out.push((
                    "error",
                    json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": e},
                    }),
                ));
            }
        }
    }
}

#[allow(dead_code)]
pub fn stream_event_name(ev: &wire::StreamEvent) -> &'static str {
    match ev {
        wire::StreamEvent::MessageStart { .. } => "message_start",
        wire::StreamEvent::ContentBlockStart { .. } => "content_block_start",
        wire::StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        wire::StreamEvent::ContentBlockStop { .. } => "content_block_stop",
        wire::StreamEvent::MessageDelta { .. } => "message_delta",
        wire::StreamEvent::MessageStop => "message_stop",
        wire::StreamEvent::Error { .. } => "error",
        wire::StreamEvent::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolCallChunk;

    fn collect(state: &mut SseState, events: Vec<LlmEvent>) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        for ev in events {
            out.extend(state.on_event(ev));
        }
        out
    }

    #[test]
    fn pure_text_turn() {
        let mut s = SseState::new("m".into());
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::Token("Hi".into()),
                LlmEvent::Token(" there".into()),
                LlmEvent::Done,
            ],
        );
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }

    #[test]
    fn signature_emitted_before_block_stop() {
        let mut s = SseState::new("m".into());
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::Reasoning("hmm".into()),
                LlmEvent::ReasoningSignature("sig-A".into()),
                LlmEvent::Token("answer".into()),
                LlmEvent::Done,
            ],
        );
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        // Expect: message_start, thinking start, thinking delta, signature delta,
        // thinking stop (lookahead-buffered), text start, text delta,
        // text stop, message_delta, message_stop
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta", // thinking_delta
                "content_block_delta", // signature_delta
                "content_block_stop",  // thinking block closed AFTER signature
                "content_block_start", // text
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        // Sanity: the third frame is the signature_delta.
        let sig = &frames[3].1["delta"];
        assert_eq!(sig["type"], "signature_delta");
        assert_eq!(sig["signature"], "sig-A");
    }

    #[test]
    fn tool_call_streamed() {
        let mut s = SseState::new("m".into());
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::ToolCallChunk(ToolCallChunk {
                    id: "tu1".into(),
                    name: "calc".into(),
                    delta: "{\"a\":".into(),
                    index: 0,
                }),
                LlmEvent::ToolCallChunk(ToolCallChunk {
                    id: "tu1".into(),
                    name: "calc".into(),
                    delta: "1}".into(),
                    index: 0,
                }),
                LlmEvent::Done,
            ],
        );
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let stop_reason = &frames[5].1["delta"]["stop_reason"];
        assert_eq!(stop_reason, "tool_use");
    }

    #[test]
    fn done_with_zero_events() {
        let mut s = SseState::new("m".into());
        let frames = collect(&mut s, vec![LlmEvent::Done]);
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["message_start", "message_delta", "message_stop"]);
    }

    #[test]
    fn synthesized_tool_call_without_chunks() {
        let mut s = SseState::new("m".into());
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::ToolCall(crate::request::ToolCall {
                    id: "tu1".into(),
                    name: "calc".into(),
                    arguments: "{\"a\":1}".into(),
                }),
                LlmEvent::Done,
            ],
        );
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }
}
