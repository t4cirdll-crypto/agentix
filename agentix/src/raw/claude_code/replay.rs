//! History-replay state for the `claude-code` intercepting proxy.
//!
//! # Why this exists
//!
//! Feeding a tool-loop turn to `claude -p` via `--resume`/stdin made the CLI
//! reshape the history every turn (a moving `"Continue from where you left
//! off."` injection landing on a different mid-history message each time),
//! which broke Anthropic's prompt cache down to a system-only hit (see
//! issue #7). The fix stops reshaping: we resume only up to the last *settled*
//! user message and let the CLI re-derive the current tool loop **live**, as a
//! single continuous session — exactly the shape interactive Claude Code
//! produces, where caching already works.
//!
//! To make "live" free and deterministic, an in-process MITM proxy
//! ([`super::proxy`]) answers the CLI's model calls for the already-known steps
//! with the recorded assistant message, and the stub MCP server answers the
//! corresponding tool calls with the recorded tool result. Once the recorded
//! steps are exhausted the proxy lets the next call hit Anthropic for the one
//! real generation.
//!
//! [`ReplayState`] is the shared coordinator between those two halves.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use crate::msg::LlmEvent;
use crate::request::{Content, Message, ToolCall};
use crate::server::anthropic::outbound::SseState;

use super::session::MCP_SERVER_NAME;

/// A recorded tool result the stub MCP server hands back during replay. Matched
/// against the live call by tool name, preferring an exact arguments match when
/// the same tool was called more than once in a turn.
struct ToolMatch {
    name: String,
    args: Value,
    result: String,
    /// Consumed once returned, so duplicate (name,args) calls in a turn still
    /// drain in order.
    taken: bool,
}

/// What the proxy should do with an incoming `POST /v1/messages` model call.
pub(crate) enum TurnAction {
    /// Answer with this pre-rendered Anthropic SSE (a recorded assistant turn).
    Fake(String),
    /// Let the request through to Anthropic for the one real generation.
    Passthrough,
    /// Recorded steps already replayed and the real turn already served — end
    /// the CLI's loop cheaply with a no-op `end_turn` instead of paying for
    /// another upstream call.
    Halt(String),
}

/// Shared state driving a single replayed tool-loop turn. Cloned (as an `Arc`)
/// into both the proxy handler and the stub MCP server.
pub(crate) struct ReplayState {
    /// Pre-rendered SSE for each recorded assistant turn, in order.
    fakes: Vec<String>,
    /// Minimal `end_turn` SSE used to halt the CLI after the real turn.
    halt: String,
    /// Per-turn consumable queue of recorded tool results.
    tool_queues: Vec<Mutex<Vec<ToolMatch>>>,
    /// Index of the next model call the proxy will see (0-based).
    next_post: AtomicUsize,
    /// Index of the turn currently executing, for MCP result matching.
    current_turn: AtomicUsize,
}

impl ReplayState {
    /// Number of faked (replayed) turns. The CLI's stdout emits this many
    /// turns before the genuine passthrough turn, so the parser skips exactly
    /// this many `message_stop`s to reach the real one.
    pub(crate) fn fake_count(&self) -> usize {
        self.fakes.len()
    }

    /// Decide what to do with the next model call. Called once per
    /// `POST /v1/messages` the proxy sees.
    pub(crate) fn next_action(&self) -> TurnAction {
        let i = self.next_post.fetch_add(1, Ordering::SeqCst);
        if i < self.fakes.len() {
            self.current_turn.store(i, Ordering::SeqCst);
            TurnAction::Fake(self.fakes[i].clone())
        } else if i == self.fakes.len() {
            TurnAction::Passthrough
        } else {
            TurnAction::Halt(self.halt.clone())
        }
    }

    /// Return the recorded result for a tool call in the current turn, matched
    /// by name (preferring an exact arguments match). `None` once the turn's
    /// recorded results are exhausted — the proxy is past replay.
    pub(crate) fn take_tool_result(&self, name: &str, args: &Value) -> Option<String> {
        let turn = self.current_turn.load(Ordering::SeqCst);
        let queue = self.tool_queues.get(turn)?;
        let mut queue = queue.lock().ok()?;
        let exact = queue
            .iter()
            .position(|m| !m.taken && m.name == name && &m.args == args);
        let pos = exact.or_else(|| queue.iter().position(|m| !m.taken && m.name == name))?;
        queue[pos].taken = true;
        Some(queue[pos].result.clone())
    }
}

/// Build replay state from the recorded portion of history — everything after
/// the last user message, i.e. a run of `Assistant`(tool_use) → `ToolResult…`
/// groups. Returns `None` when there is nothing to replay (no recorded
/// assistant turn), in which case the caller uses the plain single-shot path.
pub(crate) fn build_replay(recorded: &[Message], model: &str) -> Option<ReplayState> {
    let mut fakes: Vec<String> = Vec::new();
    let mut tool_queues: Vec<Mutex<Vec<ToolMatch>>> = Vec::new();

    let mut i = 0;
    while i < recorded.len() {
        let Message::Assistant {
            content,
            tool_calls,
            ..
        } = &recorded[i]
        else {
            // Stray non-assistant (shouldn't happen in a well-formed loop) —
            // skip it rather than desync the turn indexing.
            i += 1;
            continue;
        };

        // Gather this turn's tool results (the consecutive ToolResults that
        // follow the assistant) keyed by call id.
        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut j = i + 1;
        while let Some(Message::ToolResult { call_id, content }) = recorded.get(j) {
            results.insert(call_id.clone(), join_text(content));
            j += 1;
        }

        let (sse, matches) = render_turn(content.as_deref(), tool_calls, &results, model);
        fakes.push(sse);
        tool_queues.push(Mutex::new(matches));
        i = j;
    }

    if fakes.is_empty() {
        return None;
    }

    Some(ReplayState {
        halt: render_sse(model, vec![LlmEvent::Done]),
        fakes,
        tool_queues,
        next_post: AtomicUsize::new(0),
        current_turn: AtomicUsize::new(0),
    })
}

/// Render one recorded assistant turn to Anthropic SSE and build its tool-match
/// queue. Tool-use ids are normalised to `toolu_*` so the assistant blocks the
/// CLI rebuilds stay valid for the real passthrough request; the MCP server
/// matches on `(name, args)` so id rewriting doesn't affect result lookup.
fn render_turn(
    content: Option<&str>,
    tool_calls: &[ToolCall],
    results: &std::collections::HashMap<String, String>,
    model: &str,
) -> (String, Vec<ToolMatch>) {
    let mut events: Vec<LlmEvent> = Vec::new();
    if let Some(text) = content.filter(|t| !t.is_empty()) {
        events.push(LlmEvent::Token(text.to_string()));
    }

    let mut matches: Vec<ToolMatch> = Vec::new();
    for tc in tool_calls {
        let input: Value = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
        // The model-facing (and thus wire) name is MCP-namespaced; the MCP
        // server receives the bare name, so match on the bare name.
        events.push(LlmEvent::ToolCall(ToolCall {
            id: normalise_id(&tc.id),
            name: format!("mcp__{MCP_SERVER_NAME}__{}", tc.name),
            arguments: tc.arguments.clone(),
        }));
        matches.push(ToolMatch {
            name: tc.name.clone(),
            args: if input == Value::Null {
                serde_json::json!({})
            } else {
                input
            },
            result: results.get(&tc.id).cloned().unwrap_or_default(),
            taken: false,
        });
    }
    events.push(LlmEvent::Done);

    (render_sse(model, events), matches)
}

/// Run a list of `LlmEvent`s through the shared Anthropic SSE state machine and
/// concatenate the frames into a wire SSE body.
fn render_sse(model: &str, events: Vec<LlmEvent>) -> String {
    let mut state = SseState::new(model.to_string());
    let mut out = String::new();
    for ev in events {
        for (name, payload) in state.on_event(ev) {
            out.push_str("event: ");
            out.push_str(name);
            out.push_str("\ndata: ");
            out.push_str(&payload.to_string());
            out.push_str("\n\n");
        }
    }
    out
}

fn normalise_id(id: &str) -> String {
    if id.starts_with("toolu_") {
        id.to_string()
    } else {
        format!("toolu_{}", uuid::Uuid::new_v4().simple())
    }
}

fn join_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asst(tool: &str, id: &str, args: &str) -> Message {
        Message::Assistant {
            content: None,
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                name: tool.into(),
                arguments: args.into(),
            }],
            provider_data: None,
        }
    }
    fn tr(id: &str, text: &str) -> Message {
        Message::ToolResult {
            call_id: id.into(),
            content: vec![Content::text(text)],
        }
    }

    #[test]
    fn no_recorded_turns_returns_none() {
        assert!(build_replay(&[], "m").is_none());
    }

    #[test]
    fn replays_each_turn_then_passthrough_then_halt() {
        let recorded = vec![
            asst("bash", "c1", "{\"cmd\":\"ls\"}"),
            tr("c1", "file.txt"),
            asst("bash", "c2", "{\"cmd\":\"cat\"}"),
            tr("c2", "hello"),
        ];
        let st = build_replay(&recorded, "m").expect("two turns");

        // Turn 0
        let a0 = st.next_action();
        assert!(matches!(a0, TurnAction::Fake(_)));
        assert_eq!(
            st.take_tool_result("bash", &serde_json::json!({"cmd":"ls"}))
                .as_deref(),
            Some("file.txt")
        );
        // Turn 1
        assert!(matches!(st.next_action(), TurnAction::Fake(_)));
        assert_eq!(
            st.take_tool_result("bash", &serde_json::json!({"cmd":"cat"}))
                .as_deref(),
            Some("hello")
        );
        // Real generation
        assert!(matches!(st.next_action(), TurnAction::Passthrough));
        // Anything after halts without an upstream call.
        assert!(matches!(st.next_action(), TurnAction::Halt(_)));
        assert!(matches!(st.next_action(), TurnAction::Halt(_)));
    }

    #[test]
    fn fake_sse_is_wellformed_and_namespaced() {
        let recorded = vec![asst("bash", "c1", "{\"cmd\":\"ls\"}"), tr("c1", "ok")];
        let st = build_replay(&recorded, "m").unwrap();
        let TurnAction::Fake(sse) = st.next_action() else {
            panic!("expected fake")
        };
        assert!(sse.contains("event: message_start"));
        assert!(sse.contains("event: content_block_start"));
        assert!(sse.contains("event: message_stop"));
        // Tool surfaced to the CLI under the MCP namespace.
        assert!(sse.contains("mcp__agentix__bash"));
        // tool_use id normalised to toolu_*.
        assert!(sse.contains("toolu_"));
    }

    #[test]
    fn duplicate_tool_calls_drain_in_order() {
        let recorded = vec![
            Message::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![
                    ToolCall {
                        id: "a".into(),
                        name: "t".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "b".into(),
                        name: "t".into(),
                        arguments: "{}".into(),
                    },
                ],
                provider_data: None,
            },
            tr("a", "first"),
            tr("b", "second"),
        ];
        let st = build_replay(&recorded, "m").unwrap();
        let _ = st.next_action(); // enter turn 0
        let empty = serde_json::json!({});
        // Same (name,args) twice → drains both distinct recorded results.
        let got1 = st.take_tool_result("t", &empty).unwrap();
        let got2 = st.take_tool_result("t", &empty).unwrap();
        let mut got = vec![got1, got2];
        got.sort();
        assert_eq!(got, vec!["first".to_string(), "second".to_string()]);
        assert!(st.take_tool_result("t", &empty).is_none());
    }
}
