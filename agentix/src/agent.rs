use async_stream::stream;
use futures::StreamExt;
use tracing::{debug, warn};

use crate::Content;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::request::{Message, Request, ToolCall, truncate_to_token_budget};
use crate::tool_trait::{Tool, ToolOutput};
use crate::types::UsageStats;

// ── AgentEvent ────────────────────────────────────────────────────────────────

/// Events emitted by [`Agent::run`] over the course of a full generation loop
/// (potentially multiple LLM requests interleaved with tool executions).
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A text token from the LLM.
    Token(String),
    /// A reasoning/thinking token from the LLM.
    Reasoning(String),
    /// A streaming partial tool-call chunk (for live UI display).
    ToolCallChunk(crate::types::ToolCallChunk),
    /// A fully assembled tool call, about to be executed.
    ToolCallStart(ToolCall),
    /// Progress update from a tool (before the final result).
    ToolProgress {
        id: String,
        name: String,
        progress: String,
    },
    /// The final result of a tool execution.
    ToolResult {
        id: String,
        name: String,
        content: Vec<crate::request::Content>,
    },
    /// Token usage from one LLM request.
    Usage(UsageStats),
    /// Emitted once when the agent loop finishes normally.
    /// Contains cumulative token usage across all LLM requests in this run.
    Done(UsageStats),
    /// A recoverable stream error that was treated as end-of-stream.
    Warning(String),
    /// A fatal error — the stream will end after this.
    Error(String),
}

impl AgentEvent {
    /// Return the text content of a [`AgentEvent::ToolResult`] as a single
    /// string, joining multiple text parts with newlines and skipping images.
    /// Returns `None` for other event variants.
    pub fn text(&self) -> Option<String> {
        if let AgentEvent::ToolResult { content, .. } = self {
            let s = content
                .iter()
                .filter_map(|p| {
                    if let crate::request::Content::Text { text } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(s)
        } else {
            None
        }
    }
}

// ── agent() ───────────────────────────────────────────────────────────────────

/// Drive the LLM ↔ tool agentic loop and yield [`AgentEvent`]s.
///
/// - `tools` — the tool bundle to dispatch tool calls to
/// - `client` — HTTP client (owned, moved into the stream)
/// - `request` — base request config (system prompt, model, etc.; messages will be set per-turn)
/// - `history` — initial conversation history; truncate or summarize before passing if needed
///
/// Drop the returned stream to abort.
///
/// # Example
/// ```no_run
/// use agentix::{AgentEvent, Request, Provider, ToolBundle};
/// use futures::StreamExt;
///
/// # async fn run() {
/// let client = reqwest::Client::new();
/// let request = Request::new(Provider::OpenAI, "sk-...");
/// let mut stream = agentix::agent(ToolBundle::default(), client, request, vec![], None);
/// while let Some(event) = stream.next().await {
///     match event {
///         AgentEvent::Token(t) => print!("{t}"),
///         AgentEvent::ToolResult { ref name, ref content, .. } => {
///             let text = content.iter()
///                 .filter_map(|p| if let agentix::Content::Text { text } = p { Some(text.as_str()) } else { None })
///                 .collect::<Vec<_>>().join(" ");
///             println!("\n[{name}] → {text}");
///         }
///         AgentEvent::Error(e) => eprintln!("error: {e}"),
///         _ => {}
///     }
/// }
/// # }
/// ```
pub fn agent(
    tools: impl Tool + 'static,
    client: reqwest::Client,
    base_request: Request,
    mut history: Vec<Message>,
    history_budget: Option<usize>,
) -> futures::stream::BoxStream<'static, AgentEvent> {
    let tools: std::sync::Arc<dyn Tool> = std::sync::Arc::new(tools);

    Box::pin(stream! {
    let mut total_usage = UsageStats::default();
    loop {
            // ── Truncate history if budget set ────────────────────────
            if let Some(budget) = history_budget {
                truncate_to_token_budget(&mut history, budget);
            }

            // ── Call LLM ──────────────────────────────────────────────
            let tool_defs = tools.raw_tools();
            let req = base_request.clone()
                .messages(history.clone())
                .tools(tool_defs.clone());

            debug!(history_len = history.len(), tools = tool_defs.len(), "agent: calling LLM");
            let mut llm_stream = match req.stream(&client).await {
                Ok(s) => s,
                Err(e) => {
                    yield AgentEvent::Error(format!("LLM stream failed: {e}"));
                    return;
                }
            };

            let mut reply_buf = String::new();
            let mut reasoning_buf = String::new();
            let mut tool_calls_buf: Vec<ToolCall> = Vec::new();
            let mut provider_state: Option<serde_json::Value> = None;

            // ── Consume LLM stream ────────────────────────────────────
            loop {
                match llm_stream.next().await {
                    None | Some(LlmEvent::Done) => break,

                    Some(LlmEvent::Token(t)) => {
                        reply_buf.push_str(&t);
                        yield AgentEvent::Token(t);
                    }

                    Some(LlmEvent::Reasoning(t)) => {
                        reasoning_buf.push_str(&t);
                        yield AgentEvent::Reasoning(t);
                    }

                    Some(LlmEvent::ReasoningSignature(_)) => {
                        // Signatures are captured in AssistantState (provider_data)
                        // at end-of-turn for the agent loop's purposes.
                    }

                    Some(LlmEvent::ToolCallChunk(c)) => {
                        yield AgentEvent::ToolCallChunk(c);
                    }

                    Some(LlmEvent::ToolCall(tc)) => {
                        yield AgentEvent::ToolCallStart(tc.clone());
                        tool_calls_buf.push(tc);
                    }

                    Some(LlmEvent::AssistantState(v)) => {
                        provider_state = Some(v);
                    }

                    Some(LlmEvent::Usage(u)) => {
                        total_usage.prompt_tokens     += u.prompt_tokens;
                        total_usage.completion_tokens += u.completion_tokens;
                        total_usage.total_tokens      += u.total_tokens;
                        debug!(prompt = u.prompt_tokens, completion = u.completion_tokens, "agent: LLM usage");
                        yield AgentEvent::Usage(u);
                    }

                    Some(LlmEvent::Error(e)) => {
                        // Benign tail error (stream cut off after content arrived).
                        let benign = e.contains("Error in input stream")
                            && !reply_buf.trim().is_empty();
                        if benign {
                            warn!(error = %e, "agent: benign stream tail error");
                            yield AgentEvent::Warning(e);
                            break;
                        }
                        yield AgentEvent::Error(e);
                        return;
                    }
                }
            }

            // ── Append assistant message to history ───────────────────
            let has_reasoning = !reasoning_buf.is_empty();
            let assistant_msg = Message::Assistant {
                content: if reply_buf.is_empty() { None } else { Some(reply_buf.clone()) },
                reasoning: if has_reasoning { Some(reasoning_buf) } else { None },
                tool_calls: tool_calls_buf.clone(),
                provider_data: provider_state,
            };
            if !reply_buf.is_empty() || has_reasoning || !tool_calls_buf.is_empty() {
                history.push(assistant_msg);
            }

            // ── No tool calls → generation complete ───────────────────
            if tool_calls_buf.is_empty() {
                yield AgentEvent::Done(total_usage);
                return;
            }

            // ── Execute tools concurrently, real-time progress ────────
            // Tag each ToolOutput with its call id/name, merge all streams
            // via select_all, yield events as they arrive.
            use futures::stream::select_all;

            enum ToolMsg {
                Progress { id: String, name: String, msg: String },
                Result   { id: String, name: String, value: Vec<Content> },
            }

            debug!(count = tool_calls_buf.len(), "agent: executing tools concurrently");
            let tagged_streams: Vec<futures::stream::BoxStream<'static, ToolMsg>> =
                tool_calls_buf.iter().map(|tc| {
                    let tools = std::sync::Arc::clone(&tools);
                    let id   = tc.id.clone();
                    let name = tc.name.clone();
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                    let stream = futures::stream::once(async move {
                        tools.call(&name, args).await
                            .map(move |output| match output {
                                ToolOutput::Progress(p) =>
                                    ToolMsg::Progress { id: id.clone(), name: name.clone(), msg: p },
                                ToolOutput::Result(v) =>
                                    ToolMsg::Result { id: id.clone(), name: name.clone(), value: v },
                            })
                    }).flatten();
                    Box::pin(stream) as futures::stream::BoxStream<'static, ToolMsg>
                }).collect();

            let mut merged = select_all(tagged_streams);
            let mut tool_results: Vec<(String, Vec<crate::request::Content>)> = Vec::new();
            while let Some(msg) = merged.next().await {
                match msg {
                    ToolMsg::Progress { id, name, msg } =>
                        yield AgentEvent::ToolProgress { id, name, progress: msg },
                    ToolMsg::Result { id, name, value } => {
                        yield AgentEvent::ToolResult { id: id.clone(), name, content: value.clone() };
                        tool_results.push((id, value));
                    }
                }
            }
            for (id, content) in tool_results {
                history.push(Message::ToolResult { call_id: id, content });
            }
            // Loop back → next LLM request with tool results appended.
        }
    })
}

/// Drive the LLM ↔ tool loop using non-streaming [`Request::complete`] calls,
/// yielding one [`crate::types::CompleteResponse`] per LLM turn.
///
/// Like [`agent`] but at turn granularity instead of token granularity:
/// each item in the stream is a complete LLM response for one turn.
/// Intermediate turns (where the model called tools) are yielded before tool
/// execution; the final turn (no tool calls) is the last item.
///
/// Tool calls are executed concurrently between turns, exactly as in `agent()`.
///
/// # Usage patterns
///
/// ```no_run
/// use agentix::{ToolBundle, Request, Provider};
/// use futures::StreamExt;
///
/// # async fn run() {
/// let client = reqwest::Client::new();
/// let request = Request::new(Provider::OpenAI, "sk-...");
///
/// // Just the final text:
/// let text = agentix::agent_turns(ToolBundle::default(), client.clone(), request.clone(), vec![], None)
///     .last_content().await;
/// println!("{text}");
///
/// // Full response (with usage, tool_calls, etc.):
/// let response = agentix::agent_turns(ToolBundle::default(), client.clone(), request.clone(), vec![], None)
///     .last_ok().await;
///
/// // With per-turn progress:
/// let mut stream = agentix::agent_turns(ToolBundle::default(), client, request, vec![], None);
/// while let Some(Ok(resp)) = stream.next().await {
///     eprintln!("turn: {} tool calls", resp.tool_calls.len());
/// }
/// # }
/// ```
pub fn agent_turns(
    tools: impl Tool + 'static,
    client: reqwest::Client,
    base_request: Request,
    mut history: Vec<Message>,
    history_budget: Option<usize>,
) -> AgentTurnsStream {
    let tools: std::sync::Arc<dyn Tool> = std::sync::Arc::new(tools);

    AgentTurnsStream(Box::pin(stream! {
        loop {
            if let Some(budget) = history_budget {
                truncate_to_token_budget(&mut history, budget);
            }

            let tool_defs = tools.raw_tools();
            let req = base_request.clone()
                .messages(history.clone())
                .tools(tool_defs);

            debug!(history_len = history.len(), "agent_turns: calling LLM");

            let response = match req.complete(&client).await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let tool_calls = response.tool_calls.clone();

            history.push(Message::Assistant {
                content: response.content.clone(),
                reasoning: response.reasoning.clone(),
                tool_calls: tool_calls.clone(),
                provider_data: response.provider_data.clone(),
            });

            yield Ok(response);

            // No tool calls → final turn, stop
            if tool_calls.is_empty() {
                return;
            }

            // Execute tools concurrently
            debug!(count = tool_calls.len(), "agent_turns: executing tools");
            let futs: Vec<_> = tool_calls.iter().map(|tc| {
                let tools = std::sync::Arc::clone(&tools);
                let id   = tc.id.clone();
                let name = tc.name.clone();
                let args: serde_json::Value =
                    serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                async move {
                    let mut out = tools.call(&name, args).await;
                    let mut result: Vec<crate::request::Content> = Vec::new();
                    while let Some(o) = out.next().await {
                        if let crate::tool_trait::ToolOutput::Result(v) = o {
                            result = v;
                        }
                    }
                    (id, result)
                }
            }).collect();

            let results = futures::future::join_all(futs).await;
            for (id, content) in results {
                history.push(Message::ToolResult { call_id: id, content });
            }
            // Loop → next turn
        }
    }))
}

// ── AgentTurnsStream ──────────────────────────────────────────────────────────

/// The stream returned by [`agent_turns`].
///
/// Implements [`futures::Stream`] so you can drive it with `while let` /
/// `StreamExt` methods when you need per-turn control, and also provides
/// convenience methods for the common case of just wanting the final result.
pub struct AgentTurnsStream(
    futures::stream::BoxStream<'static, Result<crate::types::CompleteResponse, ApiError>>,
);

impl AgentTurnsStream {
    #[doc(hidden)]
    pub fn from_items(items: Vec<Result<crate::types::CompleteResponse, ApiError>>) -> Self {
        use futures::stream;
        AgentTurnsStream(Box::pin(stream::iter(items)))
    }

    /// Drain the stream and return the last successful turn's [`crate::types::CompleteResponse`],
    /// or `None` if every turn errored or the stream was empty.
    pub async fn last_ok(mut self) -> Option<crate::types::CompleteResponse> {
        let mut last = None;
        while let Some(item) = self.next().await {
            if let Ok(v) = item {
                last = Some(v);
            }
        }
        last
    }

    /// Drain the stream and return the text content of the last successful turn.
    /// Returns an empty `String` if every turn errored or the final response had no content.
    pub async fn last_content(mut self) -> String {
        let mut last = None;
        while let Some(item) = self.next().await {
            if let Ok(v) = item {
                last = Some(v);
            }
        }
        last.and_then(|r| r.content).unwrap_or_default()
    }
}

impl futures::Stream for AgentTurnsStream {
    type Item = Result<crate::types::CompleteResponse, ApiError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.0).poll_next(cx)
    }
}
