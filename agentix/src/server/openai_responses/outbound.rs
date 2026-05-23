//! `LlmEvent` stream → Responses API streaming events.
//!
//! Wire shape (one event per LlmEvent group, multiple events per item):
//!
//! ```text
//! response.created           {response: {...}}
//! response.in_progress       {response: {...}}
//!   response.output_item.added       {output_index, item: {type:"reasoning", id, summary:[]}}
//!     response.reasoning_text.delta  {output_index, delta}
//!     response.reasoning_text.done   {output_index, text}
//!   response.output_item.done        {output_index, item: {...}}
//!   response.output_item.added       {output_index, item: {type:"message", id, role:"assistant", content:[]}}
//!     response.content_part.added    {output_index, content_index, part}
//!       response.output_text.delta   {output_index, content_index, delta}
//!       response.output_text.done    {output_index, content_index, text}
//!     response.content_part.done     {output_index, content_index, part}
//!   response.output_item.done        {output_index, item: {...}}
//!   response.output_item.added       {output_index, item: {type:"function_call", id, call_id, name, arguments:""}}
//!     response.function_call_arguments.delta {output_index, delta}
//!     response.function_call_arguments.done  {output_index, arguments}
//!   response.output_item.done        {output_index, item: {...}}
//! response.completed         {response: {...full output[]...}}
//! ```
//!
//! Ordering rules:
//!   - One reasoning item per Reasoning run.
//!   - One message item per Token run; switching back to Token after a
//!     non-Token event requires closing and reopening the message.
//!   - Each function_call gets a fresh output_index.
//!   - We assemble `output[]` cumulatively and re-emit it inside
//!     `response.completed` as the canonical state.

use serde_json::{Value, json};

use crate::msg::LlmEvent;
use crate::types::{CompleteResponse, FinishReason, UsageStats};

use super::wire;

const OBJECT_RESPONSE: &str = "response";

// ── Non-streaming response builder ───────────────────────────────────────────

pub fn build_non_streaming(
    resp: CompleteResponse,
    request_model: &str,
    parent_id: Option<&str>,
    response_id: &str,
    instructions: Option<String>,
    reasoning_summary: Option<String>,
) -> Value {
    let mut output: Vec<Value> = Vec::new();

    // Prefer round-tripping the upstream Responses items if present (preserves
    // encrypted_content + IDs).
    if let Some(items) = resp
        .provider_data
        .as_ref()
        .and_then(|v| v.get("openai_responses_items"))
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
    {
        output = items.clone();
    } else {
        if let Some(reasoning) = resp.reasoning.as_deref().filter(|s| !s.is_empty()) {
            output.push(json!({
                "type": "reasoning",
                "id": format!("rs_{}", short_id()),
                "summary": [{
                    "type": "summary_text",
                    "text": reasoning,
                }],
            }));
        }
        if let Some(text) = resp.content.as_deref().filter(|t| !t.is_empty()) {
            output.push(json!({
                "type": "message",
                "id": format!("msg_{}", short_id()),
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }],
            }));
        }
        for tc in &resp.tool_calls {
            output.push(json!({
                "type": "function_call",
                "id": format!("fc_{}", short_id()),
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }));
        }
    }

    let status = match resp.finish_reason {
        FinishReason::Length => "incomplete",
        FinishReason::ContentFilter => "incomplete",
        _ => "completed",
    };

    let envelope = wire::ResponseObject {
        id: response_id.to_string(),
        object: "response",
        created_at: now_unix_seconds(),
        status: match status {
            "incomplete" => "incomplete",
            _ => "completed",
        },
        model: request_model.to_string(),
        output,
        instructions,
        tool_choice: json!("auto"),
        tools: vec![],
        previous_response_id: parent_id.map(str::to_string),
        reasoning: reasoning_summary.map(|s| json!({"effort": null, "summary": s})),
        text: Some(json!({"format": {"type": "text"}})),
        temperature: None,
        max_output_tokens: None,
        parallel_tool_calls: true,
        truncation: "disabled",
        usage: Some(wire::Usage::from(&resp.usage)),
        metadata: serde_json::Map::new(),
        incomplete_details: None,
    };
    serde_json::to_value(envelope).unwrap_or(Value::Null)
}

fn short_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

fn now_unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn synth_response_id() -> String {
    format!("resp_{}", short_id())
}

// ── Streaming state machine ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum CurrentItem {
    None,
    Reasoning {
        index: u32,
        item_id: String,
        text: String,
    },
    Message {
        index: u32,
        item_id: String,
        text: String,
    },
    FunctionCall {
        index: u32,
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

pub struct ResponsesStreamState {
    response_id: String,
    model: String,
    instructions: Option<String>,
    parent_id: Option<String>,
    reasoning_summary: Option<String>,

    response_started: bool,
    next_output_index: u32,
    current: CurrentItem,
    /// Sealed output items (in order) — replayed in `response.completed`.
    output: Vec<Value>,
    last_usage: Option<UsageStats>,
    sequence: u32,
    /// The full output items as we'd persist them — used by the server to
    /// commit to the session store after the stream finishes successfully.
    pub committed_items: Vec<Value>,
}

impl ResponsesStreamState {
    pub fn new(
        response_id: String,
        model: String,
        instructions: Option<String>,
        parent_id: Option<String>,
        reasoning_summary: Option<String>,
    ) -> Self {
        Self {
            response_id,
            model,
            instructions,
            parent_id,
            reasoning_summary,
            response_started: false,
            next_output_index: 0,
            current: CurrentItem::None,
            output: Vec::new(),
            last_usage: None,
            sequence: 0,
            committed_items: Vec::new(),
        }
    }

    fn next_seq(&mut self) -> u32 {
        let s = self.sequence;
        self.sequence += 1;
        s
    }

    fn ensure_response_started(&mut self, out: &mut Vec<(&'static str, Value)>) {
        if self.response_started {
            return;
        }
        self.response_started = true;
        let envelope = self.envelope_skeleton("in_progress");
        out.push((
            "response.created",
            json!({
                "type": "response.created",
                "sequence_number": self.next_seq(),
                "response": envelope,
            }),
        ));
        let envelope = self.envelope_skeleton("in_progress");
        out.push((
            "response.in_progress",
            json!({
                "type": "response.in_progress",
                "sequence_number": self.next_seq(),
                "response": envelope,
            }),
        ));
    }

    fn envelope_skeleton(&self, status: &'static str) -> Value {
        let envelope = wire::ResponseObject {
            id: self.response_id.clone(),
            object: OBJECT_RESPONSE,
            created_at: now_unix_seconds(),
            status,
            model: self.model.clone(),
            output: self.output.clone(),
            instructions: self.instructions.clone(),
            tool_choice: json!("auto"),
            tools: vec![],
            previous_response_id: self.parent_id.clone(),
            reasoning: self
                .reasoning_summary
                .clone()
                .map(|s| json!({"effort": null, "summary": s})),
            text: Some(json!({"format": {"type": "text"}})),
            temperature: None,
            max_output_tokens: None,
            parallel_tool_calls: true,
            truncation: "disabled",
            usage: self.last_usage.as_ref().map(wire::Usage::from),
            metadata: serde_json::Map::new(),
            incomplete_details: None,
        };
        serde_json::to_value(envelope).unwrap_or(Value::Null)
    }

    fn close_current(&mut self, out: &mut Vec<(&'static str, Value)>) {
        let cur = std::mem::replace(&mut self.current, CurrentItem::None);
        match cur {
            CurrentItem::None => {}
            CurrentItem::Reasoning {
                index,
                item_id,
                text,
            } => {
                let seq = self.next_seq();
                out.push((
                    "response.reasoning_text.done",
                    json!({
                        "type": "response.reasoning_text.done",
                        "sequence_number": seq,
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 0,
                        "text": text,
                    }),
                ));
                let item = json!({
                    "type": "reasoning",
                    "id": item_id,
                    "summary": [{
                        "type": "summary_text",
                        "text": text,
                    }],
                });
                let seq = self.next_seq();
                out.push((
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "sequence_number": seq,
                        "output_index": index,
                        "item": item.clone(),
                    }),
                ));
                self.output.push(item.clone());
                self.committed_items.push(item);
            }
            CurrentItem::Message {
                index,
                item_id,
                text,
            } => {
                let seq = self.next_seq();
                out.push((
                    "response.output_text.done",
                    json!({
                        "type": "response.output_text.done",
                        "sequence_number": seq,
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 0,
                        "text": text,
                    }),
                ));
                let part = json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                });
                let seq = self.next_seq();
                out.push((
                    "response.content_part.done",
                    json!({
                        "type": "response.content_part.done",
                        "sequence_number": seq,
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": 0,
                        "part": part.clone(),
                    }),
                ));
                let item = json!({
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "content": [part],
                    "status": "completed",
                });
                let seq = self.next_seq();
                out.push((
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "sequence_number": seq,
                        "output_index": index,
                        "item": item.clone(),
                    }),
                ));
                self.output.push(item.clone());
                self.committed_items.push(item);
            }
            CurrentItem::FunctionCall {
                index,
                item_id,
                call_id,
                name,
                arguments,
            } => {
                let seq = self.next_seq();
                out.push((
                    "response.function_call_arguments.done",
                    json!({
                        "type": "response.function_call_arguments.done",
                        "sequence_number": seq,
                        "item_id": item_id,
                        "output_index": index,
                        "arguments": arguments,
                    }),
                ));
                let item = json!({
                    "type": "function_call",
                    "id": item_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                });
                let seq = self.next_seq();
                out.push((
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "sequence_number": seq,
                        "output_index": index,
                        "item": item.clone(),
                    }),
                ));
                self.output.push(item.clone());
                self.committed_items.push(item);
            }
        }
    }

    pub fn on_event(&mut self, ev: LlmEvent) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        self.handle(ev, &mut out);
        out
    }

    fn handle(&mut self, ev: LlmEvent, out: &mut Vec<(&'static str, Value)>) {
        match ev {
            LlmEvent::Token(t) => {
                self.ensure_response_started(out);
                if !matches!(self.current, CurrentItem::Message { .. }) {
                    self.close_current(out);
                    self.open_message(out);
                }
                if t.is_empty() {
                    return;
                }
                if let CurrentItem::Message {
                    index,
                    item_id,
                    text,
                } = &mut self.current
                {
                    text.push_str(&t);
                    let seq = self.sequence;
                    self.sequence += 1;
                    out.push((
                        "response.output_text.delta",
                        json!({
                            "type": "response.output_text.delta",
                            "sequence_number": seq,
                            "item_id": item_id.clone(),
                            "output_index": *index,
                            "content_index": 0,
                            "delta": t,
                        }),
                    ));
                }
            }

            LlmEvent::Reasoning(r) => {
                self.ensure_response_started(out);
                if !matches!(self.current, CurrentItem::Reasoning { .. }) {
                    self.close_current(out);
                    self.open_reasoning(out);
                }
                if r.is_empty() {
                    return;
                }
                if let CurrentItem::Reasoning {
                    index,
                    item_id,
                    text,
                } = &mut self.current
                {
                    text.push_str(&r);
                    let seq = self.sequence;
                    self.sequence += 1;
                    out.push((
                        "response.reasoning_text.delta",
                        json!({
                            "type": "response.reasoning_text.delta",
                            "sequence_number": seq,
                            "item_id": item_id.clone(),
                            "output_index": *index,
                            "content_index": 0,
                            "delta": r,
                        }),
                    ));
                }
            }

            LlmEvent::ReasoningSignature(_) => {
                // Responses API encrypts reasoning server-side — there's no
                // signature_delta equivalent for proxied reasoning. Drop.
            }

            LlmEvent::ToolCallChunk(chunk) => {
                self.ensure_response_started(out);
                let same_call = matches!(
                    &self.current,
                    CurrentItem::FunctionCall { call_id, .. } if call_id == &chunk.id
                );
                if !same_call {
                    self.close_current(out);
                    self.open_function_call(out, chunk.id.clone(), chunk.name.clone());
                }
                if !chunk.delta.is_empty()
                    && let CurrentItem::FunctionCall {
                        index,
                        item_id,
                        arguments,
                        ..
                    } = &mut self.current
                {
                    arguments.push_str(&chunk.delta);
                    let seq = self.sequence;
                    self.sequence += 1;
                    out.push((
                        "response.function_call_arguments.delta",
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": seq,
                            "item_id": item_id.clone(),
                            "output_index": *index,
                            "delta": chunk.delta,
                        }),
                    ));
                }
            }

            LlmEvent::ToolCall(call) => {
                let already_streamed = matches!(
                    &self.current,
                    CurrentItem::FunctionCall { call_id, .. } if call_id == &call.id
                ) || self.output.iter().any(|item| {
                    item.get("type") == Some(&Value::String("function_call".into()))
                        && item.get("call_id") == Some(&Value::String(call.id.clone()))
                });
                if already_streamed {
                    return;
                }
                self.ensure_response_started(out);
                self.close_current(out);
                self.open_function_call(out, call.id.clone(), call.name.clone());
                if !call.arguments.is_empty()
                    && let CurrentItem::FunctionCall {
                        index,
                        item_id,
                        arguments,
                        ..
                    } = &mut self.current
                {
                    arguments.push_str(&call.arguments);
                    let seq = self.sequence;
                    self.sequence += 1;
                    out.push((
                        "response.function_call_arguments.delta",
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": seq,
                            "item_id": item_id.clone(),
                            "output_index": *index,
                            "delta": call.arguments,
                        }),
                    ));
                }
                self.close_current(out);
            }

            LlmEvent::Usage(u) => {
                self.last_usage = Some(u);
            }

            LlmEvent::AssistantState(_) => {
                // Anthropic-only round-trip data; not surfaced on Responses
                // wire (Responses uses its own item array round-trip via
                // openai_responses_items in provider_data).
            }

            LlmEvent::Done => {
                self.ensure_response_started(out);
                self.close_current(out);
                let envelope = self.envelope_skeleton("completed");
                let seq = self.next_seq();
                out.push((
                    "response.completed",
                    json!({
                        "type": "response.completed",
                        "sequence_number": seq,
                        "response": envelope,
                    }),
                ));
            }

            LlmEvent::Error(e) => {
                self.ensure_response_started(out);
                self.close_current(out);
                let mut envelope = self.envelope_skeleton("failed");
                if let Value::Object(map) = &mut envelope {
                    map.insert("incomplete_details".into(), json!({"reason": "error"}));
                }
                let seq = self.next_seq();
                out.push((
                    "response.failed",
                    json!({
                        "type": "response.failed",
                        "sequence_number": seq,
                        "response": envelope,
                    }),
                ));
                // Also emit a top-level error event for clients that look
                // for `event: error`.
                let seq = self.next_seq();
                out.push((
                    "error",
                    json!({
                        "type": "error",
                        "sequence_number": seq,
                        "code": "server_error",
                        "message": e,
                        "param": Value::Null,
                    }),
                ));
            }
        }
    }

    fn open_reasoning(&mut self, out: &mut Vec<(&'static str, Value)>) {
        let index = self.next_output_index;
        self.next_output_index += 1;
        let item_id = format!("rs_{}", short_id());
        let item_skel = json!({
            "type": "reasoning",
            "id": item_id,
            "summary": [],
        });
        let seq = self.next_seq();
        out.push((
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": index,
                "item": item_skel,
            }),
        ));
        self.current = CurrentItem::Reasoning {
            index,
            item_id,
            text: String::new(),
        };
    }

    fn open_message(&mut self, out: &mut Vec<(&'static str, Value)>) {
        let index = self.next_output_index;
        self.next_output_index += 1;
        let item_id = format!("msg_{}", short_id());
        let item_skel = json!({
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "content": [],
            "status": "in_progress",
        });
        let seq = self.next_seq();
        out.push((
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": index,
                "item": item_skel,
            }),
        ));
        let part = json!({
            "type": "output_text",
            "text": "",
            "annotations": [],
        });
        let seq = self.next_seq();
        out.push((
            "response.content_part.added",
            json!({
                "type": "response.content_part.added",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "part": part,
            }),
        ));
        self.current = CurrentItem::Message {
            index,
            item_id,
            text: String::new(),
        };
    }

    fn open_function_call(
        &mut self,
        out: &mut Vec<(&'static str, Value)>,
        call_id: String,
        name: String,
    ) {
        let index = self.next_output_index;
        self.next_output_index += 1;
        let item_id = format!("fc_{}", short_id());
        let item_skel = json!({
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
        });
        let seq = self.next_seq();
        out.push((
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": index,
                "item": item_skel,
            }),
        ));
        self.current = CurrentItem::FunctionCall {
            index,
            item_id,
            call_id,
            name,
            arguments: String::new(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::ToolCall;
    use crate::types::ToolCallChunk;

    fn collect(
        state: &mut ResponsesStreamState,
        events: Vec<LlmEvent>,
    ) -> Vec<(&'static str, Value)> {
        let mut out = Vec::new();
        for ev in events {
            out.extend(state.on_event(ev));
        }
        out
    }

    fn fresh_state() -> ResponsesStreamState {
        ResponsesStreamState::new("resp_test".into(), "gpt-mock".into(), None, None, None)
    }

    #[test]
    fn pure_text_emits_full_event_sequence() {
        let mut s = fresh_state();
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
                "response.created",
                "response.in_progress",
                "response.output_item.added", // message
                "response.content_part.added",
                "response.output_text.delta", // "Hi"
                "response.output_text.delta", // " there"
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let completed = frames.last().unwrap();
        let response = &completed.1["response"];
        assert_eq!(response["status"], "completed");
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["content"][0]["text"], "Hi there");
    }

    #[test]
    fn reasoning_then_text_splits_into_two_items() {
        let mut s = fresh_state();
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::Reasoning("plan".into()),
                LlmEvent::Token("answer".into()),
                LlmEvent::Done,
            ],
        );
        // Ensure both a reasoning item AND a message item land in the output.
        let response = frames.last().unwrap().1["response"].clone();
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"][0]["text"], "plan");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "answer");
    }

    #[test]
    fn tool_call_streamed_emits_function_call_events() {
        let mut s = fresh_state();
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
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"response.output_item.added"));
        assert!(names.contains(&"response.function_call_arguments.delta"));
        assert!(names.contains(&"response.function_call_arguments.done"));
        assert!(names.contains(&"response.completed"));
        let response = frames.last().unwrap().1["response"].clone();
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["call_id"], "call_1");
        assert_eq!(output[0]["name"], "calc");
        assert_eq!(output[0]["arguments"], "{\"a\":1}");
    }

    #[test]
    fn synthesized_tool_call_without_chunks() {
        let mut s = fresh_state();
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
        let response = frames.last().unwrap().1["response"].clone();
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["arguments"], "{\"a\":1}");
    }

    #[test]
    fn done_with_zero_events_emits_minimal_lifecycle() {
        let mut s = fresh_state();
        let frames = collect(&mut s, vec![LlmEvent::Done]);
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "response.created",
                "response.in_progress",
                "response.completed"
            ]
        );
        let response = frames.last().unwrap().1["response"].clone();
        let output = response["output"].as_array().unwrap();
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn error_emits_response_failed_and_error_event() {
        let mut s = fresh_state();
        let frames = collect(
            &mut s,
            vec![
                LlmEvent::Token("partial".into()),
                LlmEvent::Error("boom".into()),
            ],
        );
        let names: Vec<&str> = frames.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"response.failed"));
        assert!(names.contains(&"error"));
    }

    #[test]
    fn committed_items_match_completed_output() {
        let mut s = fresh_state();
        let _ = collect(
            &mut s,
            vec![
                LlmEvent::Reasoning("plan".into()),
                LlmEvent::Token("answer".into()),
                LlmEvent::Done,
            ],
        );
        // The committed_items vec is what the server should persist for
        // previous_response_id chaining — must include both items.
        assert_eq!(s.committed_items.len(), 2);
        assert_eq!(s.committed_items[0]["type"], "reasoning");
        assert_eq!(s.committed_items[1]["type"], "message");
    }
}
