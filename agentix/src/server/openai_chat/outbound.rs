//! Translate agentix's `LlmEvent` stream and `CompleteResponse` values back
//! into Chat Completions wire format.
//!
//! Streaming output is `chat.completion.chunk` SSE events. Compared to the
//! Anthropic SSE producer this is simpler — there are no block-start /
//! block-stop boundaries, just an open assistant message that accumulates
//! content / reasoning_content / tool_calls deltas, with one final chunk
//! carrying `finish_reason`.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::msg::LlmEvent;
use crate::types::{CompleteResponse, FinishReason, UsageStats};

use super::wire::{self, ChatCompletionChunk, ChunkChoice, Delta, DeltaFunctionCall, DeltaToolCall};

// ── Non-streaming response builder ───────────────────────────────────────────

pub fn build_response_body(resp: CompleteResponse, request_model: &str) -> Value {
    let stop_reason = stop_reason_str(&resp.finish_reason, !resp.tool_calls.is_empty());

    let tool_calls: Vec<wire::ToolCallOnMessage> = resp
        .tool_calls
        .iter()
        .map(|tc| wire::ToolCallOnMessage {
            id: tc.id.clone(),
            kind: "function".to_string(),
            function: wire::FunctionCallArgs {
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            },
        })
        .collect();

    let body = wire::ChatCompletion {
        id: synth_completion_id(),
        object: "chat.completion",
        created: now_unix_seconds(),
        model: request_model.to_string(),
        choices: vec![wire::Choice {
            index: 0,
            message: wire::ResponseMessage {
                role: "assistant",
                content: resp.content.filter(|s| !s.is_empty()),
                reasoning_content: resp.reasoning.filter(|s| !s.is_empty()),
                tool_calls,
            },
            finish_reason: Some(stop_reason.to_string()),
        }],
        usage: Some(wire::Usage::from(&resp.usage)),
    };
    serde_json::to_value(body).unwrap_or(Value::Null)
}

fn stop_reason_str(fr: &FinishReason, has_tool_calls: bool) -> &'static str {
    match fr {
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Stop | FinishReason::Other(_) => {
            if has_tool_calls { "tool_calls" } else { "stop" }
        }
    }
}

fn synth_completion_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:x}")
}

fn now_unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Streaming chunk state machine ────────────────────────────────────────────

pub struct ChunkState {
    id: String,
    model: String,
    created: u64,
    role_emitted: bool,
    /// Map from agentix tool-call id → Chat Completions slot index.
    tool_slot_by_id: HashMap<String, u32>,
    next_tool_slot: u32,
    /// Slots for which we've already emitted the `id`/`type`/`name` skeleton —
    /// subsequent argument chunks omit those fields.
    tool_skeleton_emitted: HashMap<u32, bool>,
    has_tool_calls: bool,
    last_usage: Option<UsageStats>,
    include_usage: bool,
}

impl ChunkState {
    pub fn new(model: String, include_usage: bool) -> Self {
        Self {
            id: synth_completion_id(),
            model,
            created: now_unix_seconds(),
            role_emitted: false,
            tool_slot_by_id: HashMap::new(),
            next_tool_slot: 0,
            tool_skeleton_emitted: HashMap::new(),
            has_tool_calls: false,
            last_usage: None,
            include_usage,
        }
    }

    fn skeleton(&self, choice_delta: Delta, finish_reason: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: choice_delta,
                finish_reason: finish_reason.map(|s| s.to_string()),
            }],
            usage: None,
        }
    }

    /// Emit one or more SSE frames for the given `LlmEvent`. Each frame is
    /// `(event_name, payload)`. For Chat Completions the event name is unset
    /// on the wire (just `data: ...`), so we always emit an empty event name
    /// — the `Sse` adapter treats empty event name as "no event:" line.
    pub fn on_event(&mut self, ev: LlmEvent) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        self.handle(ev, &mut out);
        out
    }

    fn ensure_role(&mut self, out: &mut Vec<(&'static str, Value)>) {
        if !self.role_emitted {
            self.role_emitted = true;
            let chunk = self.skeleton(
                Delta {
                    role: Some("assistant"),
                    ..Default::default()
                },
                None,
            );
            push_chunk(out, &chunk);
        }
    }

    fn slot_for(&mut self, id: &str) -> (u32, bool) {
        if let Some(slot) = self.tool_slot_by_id.get(id) {
            return (*slot, false);
        }
        let slot = self.next_tool_slot;
        self.next_tool_slot += 1;
        self.tool_slot_by_id.insert(id.to_string(), slot);
        self.has_tool_calls = true;
        (slot, true)
    }

    fn handle(&mut self, ev: LlmEvent, out: &mut Vec<(&'static str, Value)>) {
        match ev {
            LlmEvent::Token(t) => {
                self.ensure_role(out);
                if t.is_empty() {
                    return;
                }
                let chunk = self.skeleton(
                    Delta {
                        content: Some(t),
                        ..Default::default()
                    },
                    None,
                );
                push_chunk(out, &chunk);
            }

            LlmEvent::Reasoning(r) => {
                self.ensure_role(out);
                if r.is_empty() {
                    return;
                }
                let chunk = self.skeleton(
                    Delta {
                        reasoning_content: Some(r),
                        ..Default::default()
                    },
                    None,
                );
                push_chunk(out, &chunk);
            }

            LlmEvent::ReasoningSignature(_) => {
                // Chat Completions has no field for thinking signatures.
                // Drop silently.
            }

            LlmEvent::ToolCallChunk(chunk) => {
                self.ensure_role(out);
                let (slot, is_first) = self.slot_for(&chunk.id);
                let send_skeleton = is_first
                    || !*self
                        .tool_skeleton_emitted
                        .get(&slot)
                        .unwrap_or(&false);
                let delta_tc = if send_skeleton {
                    self.tool_skeleton_emitted.insert(slot, true);
                    DeltaToolCall {
                        index: slot,
                        id: Some(chunk.id),
                        kind: Some("function"),
                        function: Some(DeltaFunctionCall {
                            name: Some(chunk.name),
                            arguments: if chunk.delta.is_empty() {
                                None
                            } else {
                                Some(chunk.delta)
                            },
                        }),
                    }
                } else if chunk.delta.is_empty() {
                    return;
                } else {
                    DeltaToolCall {
                        index: slot,
                        id: None,
                        kind: None,
                        function: Some(DeltaFunctionCall {
                            name: None,
                            arguments: Some(chunk.delta),
                        }),
                    }
                };
                let chunk_msg = self.skeleton(
                    Delta {
                        tool_calls: vec![delta_tc],
                        ..Default::default()
                    },
                    None,
                );
                push_chunk(out, &chunk_msg);
            }

            LlmEvent::ToolCall(call) => {
                if self.tool_slot_by_id.contains_key(&call.id) {
                    // Already streamed via chunks; nothing more to send.
                    return;
                }
                self.ensure_role(out);
                let (slot, _) = self.slot_for(&call.id);
                self.tool_skeleton_emitted.insert(slot, true);
                let delta_tc = DeltaToolCall {
                    index: slot,
                    id: Some(call.id),
                    kind: Some("function"),
                    function: Some(DeltaFunctionCall {
                        name: Some(call.name),
                        arguments: if call.arguments.is_empty() {
                            None
                        } else {
                            Some(call.arguments)
                        },
                    }),
                };
                let chunk_msg = self.skeleton(
                    Delta {
                        tool_calls: vec![delta_tc],
                        ..Default::default()
                    },
                    None,
                );
                push_chunk(out, &chunk_msg);
            }

            LlmEvent::Usage(u) => {
                self.last_usage = Some(u);
            }

            LlmEvent::AssistantState(_) => {
                // Anthropic-specific reasoning signatures; not surfaced on
                // Chat Completions wire.
            }

            LlmEvent::Done => {
                self.ensure_role(out);
                let stop = if self.has_tool_calls { "tool_calls" } else { "stop" };
                let final_chunk = self.skeleton(Delta::default(), Some(stop));
                push_chunk(out, &final_chunk);

                if self.include_usage {
                    let usage = self.last_usage.clone().unwrap_or_default();
                    let usage_chunk = ChatCompletionChunk {
                        id: self.id.clone(),
                        object: "chat.completion.chunk",
                        created: self.created,
                        model: self.model.clone(),
                        choices: vec![],
                        usage: Some(wire::Usage::from(&usage)),
                    };
                    push_chunk(out, &usage_chunk);
                }
                out.push(("", Value::String("[DONE]".to_string())));
            }

            LlmEvent::Error(e) => {
                // OpenAI streams emit errors as a JSON frame with the standard
                // error envelope, then close the connection (no [DONE]).
                let payload = json!({
                    "error": {
                        "message": e,
                        "type": "server_error",
                        "param": Value::Null,
                        "code": Value::Null,
                    }
                });
                out.push(("", payload));
            }
        }
    }
}

fn push_chunk(out: &mut Vec<(&'static str, Value)>, chunk: &ChatCompletionChunk) {
    let v = serde_json::to_value(chunk).unwrap_or(Value::Null);
    out.push(("", v));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::ToolCall;
    use crate::types::ToolCallChunk;

    fn collect(state: &mut ChunkState, events: Vec<LlmEvent>) -> Vec<Value> {
        let mut out = Vec::new();
        for ev in events {
            for (_, payload) in state.on_event(ev) {
                out.push(payload);
            }
        }
        out
    }

    fn deltas_of(frames: &[Value]) -> Vec<&Value> {
        frames
            .iter()
            .filter(|v| v.is_object())
            .filter_map(|v| v.get("choices")?.as_array()?.first()?.get("delta"))
            .collect()
    }

    #[test]
    fn pure_text_turn() {
        let mut s = ChunkState::new("m".into(), false);
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::Token("Hi".into()),
                LlmEvent::Token(" there".into()),
                LlmEvent::Done,
            ],
        );
        // role chunk, two content deltas, finish chunk, [DONE]
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(frames[1]["choices"][0]["delta"]["content"], "Hi");
        assert_eq!(frames[2]["choices"][0]["delta"]["content"], " there");
        assert_eq!(frames[3]["choices"][0]["finish_reason"], "stop");
        assert_eq!(frames[4], Value::String("[DONE]".into()));
    }

    #[test]
    fn reasoning_then_text() {
        let mut s = ChunkState::new("m".into(), false);
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::Reasoning("thinking".into()),
                LlmEvent::Token("answer".into()),
                LlmEvent::Done,
            ],
        );
        let deltas = deltas_of(&frames);
        // role, reasoning, content, finish (finish has empty delta)
        assert!(deltas[1]["reasoning_content"] == "thinking");
        assert!(deltas[2]["content"] == "answer");
    }

    #[test]
    fn tool_call_streamed_includes_slot_index_and_finish_reason() {
        let mut s = ChunkState::new("m".into(), false);
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::ToolCallChunk(ToolCallChunk {
                    id: "call_1".into(),
                    name: "calc".into(),
                    delta: "{\"a\":".into(),
                    index: 0,
                }),
                LlmEvent::ToolCallChunk(ToolCallChunk {
                    id: "call_1".into(),
                    name: "calc".into(),
                    delta: "1}".into(),
                    index: 0,
                }),
                LlmEvent::Done,
            ],
        );
        // role, first tool chunk (with id+name+args), second tool chunk (args only), finish
        let first_tc = &frames[1]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(first_tc["index"], 0);
        assert_eq!(first_tc["id"], "call_1");
        assert_eq!(first_tc["function"]["name"], "calc");
        assert_eq!(first_tc["function"]["arguments"], "{\"a\":");
        let second_tc = &frames[2]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(second_tc["index"], 0);
        assert!(second_tc["id"].is_null(), "second chunk must not repeat id");
        assert!(
            second_tc["function"]["name"].is_null(),
            "second chunk must not repeat name"
        );
        assert_eq!(second_tc["function"]["arguments"], "1}");
        assert_eq!(frames[3]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn synthesized_tool_call_without_chunks() {
        let mut s = ChunkState::new("m".into(), false);
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::ToolCall(ToolCall {
                    id: "call_1".into(),
                    name: "calc".into(),
                    arguments: "{\"a\":1}".into(),
                }),
                LlmEvent::Done,
            ],
        );
        let tc = &frames[1]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["id"], "call_1");
        assert_eq!(tc["function"]["arguments"], "{\"a\":1}");
        assert_eq!(frames[2]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn include_usage_appends_usage_chunk() {
        let mut s = ChunkState::new("m".into(), true);
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::Token("hi".into()),
                LlmEvent::Usage(UsageStats {
                    prompt_tokens: 5,
                    completion_tokens: 2,
                    total_tokens: 7,
                    ..Default::default()
                }),
                LlmEvent::Done,
            ],
        );
        // role, content, finish, usage chunk, [DONE]
        assert_eq!(frames.len(), 5);
        let usage = &frames[3]["usage"];
        assert_eq!(usage["prompt_tokens"], 5);
        assert_eq!(usage["completion_tokens"], 2);
        assert_eq!(frames[4], Value::String("[DONE]".into()));
    }

    #[test]
    fn done_with_zero_events_emits_role_and_finish() {
        let mut s = ChunkState::new("m".into(), false);
        let frames = collect(&mut s, vec![LlmEvent::Done]);
        // role chunk, finish chunk, [DONE]
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(frames[1]["choices"][0]["finish_reason"], "stop");
    }
}
