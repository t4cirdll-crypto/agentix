//! Raw `claude-code` provider — drives `claude -p` as a single-turn LLM and
//! emits `LlmEvent`s / returns a `CompleteResponse`, matching every other
//! raw provider in this crate.
//!
//! # How it works
//!
//! 1. Spawn an in-process MCP server whose tools are schema-only **stubs**
//!    (see [`StubTools`]). The caller's [`ToolDefinition`]s are surfaced so
//!    the model can emit `tool_use` blocks, but `call()` returns an empty
//!    result instantly — the caller dispatches tool calls externally.
//! 2. Spawn `claude -p --input-format stream-json --output-format stream-json`
//!    connected to that MCP server over loopback HTTP.
//! 3. Feed the last user message on stdin, parse stream-json lines on stdout.
//! 4. On the first `assistant` message, flush final `ToolCall`s + `Usage`,
//!    yield `Done`, and kill the subprocess. Further turns (tool dispatch,
//!    follow-up) are the caller's responsibility.
//!
//! Auth comes from the local `claude` CLI (Max OAuth / keychain); `api_key`
//! is ignored.

pub(crate) mod proxy;
pub(crate) mod replay;
pub(crate) mod session;

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tracing::{debug, info, warn};

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::mcp_server::McpServer;
use crate::msg::LlmEvent;
use crate::raw::shared::ToolDefinition;
use crate::request::{Message, ToolCall};
use crate::tool_trait::{Tool, ToolOutput};
use crate::types::{CompleteResponse, FinishReason, PartialToolCall, ToolCallChunk, UsageStats};

use self::replay::ReplayState;
use self::session::{
    Cleanup, MCP_SERVER_NAME, is_tool_result_content, parse_usage, split_last_user,
    strip_mcp_prefix, write_fake_session,
};

fn ensure_toolu_id(id: &str, id_map: &mut HashMap<String, String>) -> String {
    if id.starts_with("toolu_") {
        return id.to_string();
    }
    if let Some(mapped) = id_map.get(id) {
        return mapped.clone();
    }
    let mapped = format!("toolu_{}", uuid::Uuid::new_v4().simple());
    id_map.insert(id.to_string(), mapped.clone());
    mapped
}

fn assistant_replay_message(
    assistant: Message,
    session_id: Option<&str>,
) -> Option<(serde_json::Value, HashMap<String, String>)> {
    let Message::Assistant {
        content,
        reasoning: _,
        tool_calls,
        provider_data,
    } = assistant
    else {
        return None;
    };

    let mut id_map = HashMap::new();

    // Preferred path: a previous turn captured the raw `anthropic_content`
    // blocks (incl. thinking + signature) into `provider_data`. Replay them
    // verbatim so signatures stay valid. Anthropic hashes (model, content)
    // and re-validates on submission; any byte mutation breaks the chain.
    // tool_use ids in captured blocks are already `toolu_*`, so id_map stays
    // empty and `remap_tool_use_ids` is a no-op for the matching tool_results.
    let blocks: Vec<serde_json::Value> = if let Some(raw_blocks) = provider_data
        .as_ref()
        .and_then(|pd| pd.get("anthropic_content"))
        .and_then(|x| x.as_array())
        .filter(|arr| !arr.is_empty())
    {
        raw_blocks.clone()
    } else {
        // Fallback: reconstruct from `content` + `tool_calls`. This loses any
        // thinking blocks, so the model starts a fresh chain-of-thought next
        // turn — but it's signature-safe.
        let mut blocks = Vec::new();
        if let Some(text) = content
            && !text.is_empty()
        {
            blocks.push(serde_json::json!({"type": "text", "text": text}));
        }
        for tc in tool_calls {
            let id = ensure_toolu_id(&tc.id, &mut id_map);
            let input: serde_json::Value =
                serde_json::from_str(&tc.arguments).unwrap_or_else(|_| serde_json::json!({}));
            blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": format!("mcp__{}__{}", MCP_SERVER_NAME, tc.name),
                "input": input,
                "caller": {"type": "direct"},
            }));
        }
        blocks
    };

    Some((
        serde_json::json!({
            "type": "assistant",
            "session_id": session_id.unwrap_or(""),
            "parent_tool_use_id": null,
            "uuid": uuid::Uuid::new_v4().to_string(),
            "message": {
                "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "type": "message",
                "role": "assistant",
                "content": blocks,
                "model": "claude-code",
                "stop_reason": "tool_use",
                "stop_sequence": null,
                "stop_details": null,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        }),
        id_map,
    ))
}

// ── Stub tools ───────────────────────────────────────────────────────────────

/// Surfaces caller-provided tool schemas to claude without executing anything.
///
/// We kill the subprocess on the first `assistant` message, but that kill is
/// asynchronous — claude may still hit the MCP server before SIGKILL lands.
/// Returning an empty result instantly prevents a blocked tool-call response
/// from pinning the subprocess alive until our drop cleanup.
struct StubTools {
    defs: Vec<ToolDefinition>,
    /// On the tool-loop replay path, the stub returns *recorded* tool results
    /// (so the CLI rebuilds the exact history) instead of an empty stub.
    replay: Option<Arc<ReplayState>>,
}

#[async_trait]
impl Tool for StubTools {
    fn raw_tools(&self) -> Vec<ToolDefinition> {
        self.defs.clone()
    }
    async fn call(&self, name: &str, args: serde_json::Value) -> BoxStream<'static, ToolOutput> {
        if let Some(state) = &self.replay
            && let Some(text) = state.take_tool_result(name, &args)
        {
            return futures::stream::iter(vec![ToolOutput::Result(vec![
                crate::request::Content::text(text),
            ])])
            .boxed();
        }
        futures::stream::iter(vec![ToolOutput::Result(vec![])]).boxed()
    }
}

// ── Subprocess setup (shared) ────────────────────────────────────────────────

/// A spawned `claude -p` subprocess plus its cleanup guard.
struct Started {
    guard: Cleanup,
    child: Child,
    /// `Some(n)` on the tool-loop replay path: the genuine turn is the turn
    /// after `n` faked replay turns on the CLI's own stdout, so the caller
    /// parses stdout and skips that many `message_stop`s to reach it.
    skip_turns: Option<usize>,
}

/// The replay plan for a tool-loop turn: resume up to the last settled user
/// message and re-derive everything after it live via the proxy + MCP.
struct ReplayPlan {
    /// History up to (but excluding) the last user message — resumed verbatim.
    resume: Vec<Message>,
    /// The last user message — fed on stdin to start the live run.
    trigger: Vec<crate::request::Content>,
    /// Shared replay coordinator (proxy fakes assistant turns, MCP returns the
    /// recorded tool results).
    state: std::sync::Arc<ReplayState>,
}

/// Decide whether this turn takes the replay path. It does exactly when the
/// history tail is a `tool_result` (a tool-loop turn) and there is a settled
/// user message to resume up to — the case where the old resume/stdin split
/// collapsed the prompt cache (issue #7). A fresh user turn returns `None` and
/// uses the plain single-shot path.
fn plan_replay(messages: &[Message], model: &str) -> Option<ReplayPlan> {
    if !matches!(messages.last(), Some(Message::ToolResult { .. })) {
        return None;
    }
    let last_user = messages.iter().rposition(|m| matches!(m, Message::User(_)))?;
    let state = replay::build_replay(&messages[last_user + 1..], model)?;
    let Message::User(parts) = &messages[last_user] else {
        return None;
    };
    Some(ReplayPlan {
        resume: messages[..last_user].to_vec(),
        trigger: parts.clone(),
        state: std::sync::Arc::new(state),
    })
}

/// Build the MCP server, write the config + fake session files, optionally spin
/// up the intercepting replay proxy, spawn `claude -p`, and feed the user
/// message on stdin. The caller drives the returned [`Started`] and is
/// responsible for `drop(child)` (SIGKILL via `kill_on_drop`) then
/// `drop(guard)` (abort MCP/proxy tasks + remove temp files), **in that order**.
async fn start_claude(
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<Started, ApiError> {
    let replay_plan = plan_replay(messages, &config.model);

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let mcp_addr = listener.local_addr()?;
    let replay_state: Option<Arc<ReplayState>> = replay_plan.as_ref().map(|p| p.state.clone());
    let stub = StubTools {
        defs: tools.to_vec(),
        replay: replay_state,
    };
    let router = McpServer::new(stub).into_axum_router();
    let mcp_task = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });

    let mut guard = Cleanup::new(mcp_task);

    let mcp_config_path =
        std::env::temp_dir().join(format!("agentix-mcp-{}.json", uuid::Uuid::new_v4()));
    let mcp_config = serde_json::json!({
        "mcpServers": {
            MCP_SERVER_NAME: {
                "type": "http",
                "url": format!("http://{}/", mcp_addr),
            }
        }
    });
    tokio::fs::write(&mcp_config_path, mcp_config.to_string())
        .await
        .map_err(|e| ApiError::Other(format!("write mcp-config: {e}")))?;
    guard.temp_files.push(mcp_config_path.clone());

    let mut resume_args: Vec<String> = Vec::new();
    let mut session_id: Option<String> = None;
    let mut stdin_prefix: Vec<serde_json::Value> = Vec::new();
    let mut tail_is_tool_result = false;
    let last_user_content: serde_json::Value;
    let mut skip_turns: Option<usize> = None;
    // (proxy listener addr, CA cert temp path) — set on the replay path.
    let mut proxy_env: Option<(std::net::SocketAddr, std::path::PathBuf)> = None;

    if let Some(plan) = replay_plan {
        // ── Replay path ──────────────────────────────────────────────────
        // Resume the settled prefix, feed the last user message on stdin, and
        // let the CLI rebuild the tool loop live against the proxy + MCP.
        last_user_content = self::session::user_content_to_json(&plan.trigger);
        if !plan.resume.is_empty() {
            let (sid, path, _id_map) = write_fake_session(&plan.resume)
                .await
                .map_err(|e| ApiError::Other(format!("write fake session: {e}")))?;
            guard.temp_files.push(path);
            resume_args.push("--resume".into());
            resume_args.push(sid.clone());
            session_id = Some(sid);
        }

        let skip = plan.state.fake_count();
        let handle = proxy::spawn_proxy(plan.state)
            .await
            .map_err(|e| ApiError::Other(format!("spawn replay proxy: {e}")))?;
        guard.proxy_task = Some(handle.task);
        let ca_path =
            std::env::temp_dir().join(format!("agentix-cc-ca-{}.pem", uuid::Uuid::new_v4()));
        tokio::fs::write(&ca_path, handle.ca_pem)
            .await
            .map_err(|e| ApiError::Other(format!("write replay CA: {e}")))?;
        guard.temp_files.push(ca_path.clone());
        proxy_env = Some((handle.addr, ca_path));
        skip_turns = Some(skip);
    } else {
        // ── Single-shot path (fresh user turn / degenerate histories) ─────
        let (mut prev_history, mut content) =
            split_last_user(messages.to_vec()).map_err(ApiError::Other)?;
        tail_is_tool_result = is_tool_result_content(&content);
        let pending_assistant = if tail_is_tool_result {
            match prev_history.last() {
                Some(Message::Assistant { .. }) => prev_history.pop(),
                _ => None,
            }
        } else {
            None
        };
        let resume_history = prev_history;
        if !resume_history.is_empty() {
            let (sid, path, id_map) = write_fake_session(&resume_history)
                .await
                .map_err(|e| ApiError::Other(format!("write fake session: {e}")))?;
            guard.temp_files.push(path);
            resume_args.push("--resume".into());
            resume_args.push(sid.clone());
            session_id = Some(sid);
            // Rewrite any tool_use_ids in the stdin message to match the
            // remapped ids in the resumed session.
            self::session::remap_tool_use_ids(&mut content, &id_map);
        }
        if let Some(assistant) = pending_assistant
            && let Some((msg, id_map)) = assistant_replay_message(assistant, session_id.as_deref())
        {
            self::session::remap_tool_use_ids(&mut content, &id_map);
            stdin_prefix.push(msg);
        }
        last_user_content = content;
    }

    let mut args: Vec<String> = vec![
        "-p".into(),
        "--strict-mcp-config".into(),
        "--mcp-config".into(),
        mcp_config_path.to_string_lossy().into_owned(),
        "--tools".into(),
        String::new(),
        "--output-format".into(),
        "stream-json".into(),
        "--input-format".into(),
        "stream-json".into(),
        "--include-partial-messages".into(),
        "--verbose".into(),
        "--permission-mode".into(),
        "bypassPermissions".into(),
        "--no-session-persistence".into(),
    ];
    if let Some(sp) = &config.system_prompt {
        args.push("--system-prompt".into());
        args.push(sp.clone());
    }
    if !config.model.is_empty() {
        args.push("--model".into());
        args.push(config.model.clone());
    }
    args.extend(resume_args);

    info!(args_len = args.len(), "spawning claude-code");
    debug!(?args, "claude-code argv");

    let mut cmd = Command::new("claude");
    cmd.args(&args)
        .env("IS_SANDBOX", "1")
        // claude CLI 2.x enables dynamic tool loading ("tool search") by default,
        // which moves MCP tools into a deferred set the model must load via the
        // built-in `ToolSearch` tool instead of presenting them directly. Combined
        // with our `--tools ""` (which disables built-ins, including ToolSearch),
        // the model can't reach the loopback tools at all and fabricates
        // `<tool_call>` text instead of emitting real `tool_use` blocks. Forcing
        // this off presents the MCP tools to the model directly, as agentix expects.
        .env("ENABLE_TOOL_SEARCH", "false")
        // We drive `claude -p` as a single-shot LLM (`--no-session-persistence`,
        // killed after the first assistant message), so its background "non-essential"
        // traffic — notably a haiku session-title prefetch on every turn — is pure
        // waste. Suppress it.
        .env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // On the replay path, route the CLI's API traffic through our intercepting
    // proxy (which still targets the real api.anthropic.com, so Max-OAuth is
    // sent normally) and trust its throwaway CA.
    if let Some((addr, ca_path)) = &proxy_env {
        let url = format!("http://{addr}");
        cmd.env("HTTP_PROXY", &url)
            .env("HTTPS_PROXY", &url)
            .env("http_proxy", &url)
            .env("https_proxy", &url)
            .env("NODE_EXTRA_CA_CERTS", ca_path)
            .env("NO_PROXY", "")
            .env("no_proxy", "");
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| ApiError::Other(format!("spawn claude: {e}")))?;

    let cc_debug = std::env::var("AGENTIX_CC_DEBUG").is_ok();
    if cc_debug {
        eprintln!("[cc-argv] claude {}", args.join(" "));
    }

    if let Some(mut stdin) = child.stdin.take() {
        for msg in stdin_prefix {
            let mut line = msg.to_string();
            line.push('\n');
            if cc_debug {
                eprintln!("[cc-stdin-replay] {line}");
            }
            if let Err(e) = stdin.write_all(line.as_bytes()).await {
                warn!(error = %e, "write stdin replay");
            }
        }
        let mut msg = serde_json::json!({
            "type": "user",
            "session_id": session_id.as_deref().unwrap_or(""),
            "parent_tool_use_id": null,
            "message": {
                "role": "user",
                "content": last_user_content,
            }
        });
        if tail_is_tool_result && let Some(obj) = msg.as_object_mut() {
            obj.insert(
                "uuid".into(),
                serde_json::Value::String(uuid::Uuid::new_v4().to_string()),
            );
            obj.insert(
                "timestamp".into(),
                serde_json::Value::String(self::session::chrono_like_now()),
            );
        }
        let mut line = msg.to_string();
        line.push('\n');
        if cc_debug {
            eprintln!("[cc-stdin-user] {line}");
        }
        if let Err(e) = stdin.write_all(line.as_bytes()).await {
            warn!(error = %e, "write stdin");
        }
        drop(stdin);
    }

    if let Some(err) = child.stderr.take() {
        tokio::spawn(async move {
            let mut elines = BufReader::new(err).lines();
            while let Ok(Some(l)) = elines.next_line().await {
                if cc_debug {
                    eprintln!("[cc-stderr] {l}");
                }
                warn!(target: "claude_code_stderr", "{}", l);
            }
        });
    }

    Ok(Started {
        guard,
        child,
        skip_turns,
    })
}

// ── Stream-JSON → LlmEvent (partial deltas) ─────────────────────────────────

#[derive(Default)]
struct StreamState {
    tool_bufs: Vec<Option<PartialToolCall>>,
    /// Raw JSON of each content block, indexed by content-block index, kept
    /// **verbatim** as it streams in (incl. fields like `caller` that claude
    /// emits but our struct types don't model). Mutated in place by deltas.
    /// On `content_block_stop` for tool_use blocks the partial input-JSON
    /// string from `tool_bufs[idx].arguments` is parsed and placed at
    /// `block_bufs[idx]["input"]`. At `message_delta` these compose the
    /// `anthropic_content` envelope shipped via `LlmEvent::AssistantState`
    /// for round-tripping thinking signatures across turns.
    block_bufs: Vec<Option<serde_json::Value>>,
}

/// Outcome of processing one stream-json line.
#[derive(Default)]
struct LineOutcome {
    events: Vec<LlmEvent>,
    /// `true` once we have seen a `message_delta` with a `stop_reason` —
    /// the canonical end-of-turn signal in claude's stream-json output.
    /// Final usage is on `event.usage`; ToolCalls have already been flushed
    /// from `content_block_stop` for their respective indices.
    turn_done: bool,
}

fn translate_stream_event_line(v: &serde_json::Value, state: &mut StreamState) -> LineOutcome {
    let mut out = LineOutcome::default();
    let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
    if ty != "stream_event" {
        return out;
    }
    let ev = match v.get("event") {
        Some(e) => e,
        None => return out,
    };
    let ety = ev.get("type").and_then(|x| x.as_str()).unwrap_or("");

    match ety {
        "content_block_start" => {
            let idx = ev.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let block = match ev.get("content_block") {
                Some(b) => b,
                None => return out,
            };
            // Capture the entire content_block JSON verbatim — `caller`,
            // model-private fields, anything else claude emits — so the round-
            // trip is byte-faithful and signatures stay valid.
            if state.block_bufs.len() <= idx {
                state.block_bufs.resize_with(idx + 1, || None);
            }
            state.block_bufs[idx] = Some(block.clone());

            if block.get("type").and_then(|x| x.as_str()) == Some("tool_use") {
                let id = block
                    .get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let raw_name = block
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = strip_mcp_prefix(&raw_name);
                if state.tool_bufs.len() <= idx {
                    state.tool_bufs.resize_with(idx + 1, || None);
                }
                state.tool_bufs[idx] = Some(PartialToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                });
                out.events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                    id,
                    name,
                    delta: String::new(),
                    index: idx as u32,
                }));
            }
        }
        "content_block_delta" => {
            let idx = ev.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let delta = match ev.get("delta") {
                Some(d) => d,
                None => return out,
            };
            match delta.get("type").and_then(|x| x.as_str()).unwrap_or("") {
                "text_delta" => {
                    if let Some(t) = delta.get("text").and_then(|x| x.as_str())
                        && !t.is_empty()
                    {
                        // Mirror into the raw-block JSON so the verbatim copy
                        // we emit at message_delta has the full text.
                        if let Some(Some(slot)) = state.block_bufs.get_mut(idx)
                            && let Some(field) = slot
                                .get_mut("text")
                                .and_then(|x| x.as_str().map(str::to_string))
                        {
                            slot["text"] = serde_json::Value::String(field + t);
                        }
                        out.events.push(LlmEvent::Token(t.to_string()));
                    }
                }
                "thinking_delta" => {
                    if let Some(t) = delta.get("thinking").and_then(|x| x.as_str())
                        && !t.is_empty()
                    {
                        if let Some(Some(slot)) = state.block_bufs.get_mut(idx)
                            && let Some(field) = slot
                                .get_mut("thinking")
                                .and_then(|x| x.as_str().map(str::to_string))
                        {
                            slot["thinking"] = serde_json::Value::String(field + t);
                        }
                        out.events.push(LlmEvent::Reasoning(t.to_string()));
                    }
                }
                "signature_delta" => {
                    // Append claude's signature for this thinking block. Anthropic
                    // hashes (model_id, thinking_text) → signature; round-tripping
                    // it verbatim lets the next turn validate without re-thinking.
                    if let Some(sig) = delta.get("signature").and_then(|x| x.as_str())
                        && !sig.is_empty()
                        && let Some(Some(slot)) = state.block_bufs.get_mut(idx)
                    {
                        let cur = slot
                            .get("signature")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        slot["signature"] = serde_json::Value::String(cur + sig);
                    }
                }
                "input_json_delta" => {
                    if let Some(partial_json) = delta.get("partial_json").and_then(|x| x.as_str())
                        && !partial_json.is_empty()
                        && let Some(Some(partial)) = state.tool_bufs.get_mut(idx)
                    {
                        partial.arguments.push_str(partial_json);
                        out.events.push(LlmEvent::ToolCallChunk(ToolCallChunk {
                            id: partial.id.clone(),
                            name: partial.name.clone(),
                            delta: partial_json.to_string(),
                            index: idx as u32,
                        }));
                    }
                }
                _ => {}
            }
        }
        "content_block_stop" => {
            let idx = ev.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            // For tool_use: parse the buffered partial-JSON string into a
            // proper JSON object and slot it into `block_bufs[idx]["input"]`,
            // then emit the final ToolCall event.
            if let Some(slot) = state.tool_bufs.get_mut(idx)
                && let Some(partial) = slot.take()
            {
                let arguments = if partial.arguments.is_empty() {
                    "{}".to_string()
                } else {
                    partial.arguments
                };
                if let Some(Some(block)) = state.block_bufs.get_mut(idx) {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&arguments).unwrap_or_else(|_| serde_json::json!({}));
                    block["input"] = parsed;
                }
                out.events.push(LlmEvent::ToolCall(ToolCall {
                    id: partial.id,
                    name: partial.name,
                    arguments,
                }));
            }
        }
        "message_delta" => {
            // End-of-turn marker. `event.usage` carries the FINAL counts for
            // this turn — `assistant`-snapshot usages are running estimates.
            if let Some(u) = ev.get("usage") {
                out.events.push(LlmEvent::Usage(parse_usage(u)));
            }

            // Bundle the raw block JSON into the `anthropic_content` envelope
            // (same wire shape as the anthropic provider) and ship it out as
            // AssistantState — gated on thinking+tool_use, the combination
            // where Anthropic enforces signature round-trip on the next turn.
            // Pure thinking-no-tools or pure text turns don't need it.
            let blocks: Vec<serde_json::Value> =
                state.block_bufs.iter().flatten().cloned().collect();
            let has_thinking = blocks.iter().any(|b| {
                matches!(
                    b.get("type").and_then(|x| x.as_str()),
                    Some("thinking") | Some("redacted_thinking")
                )
            });
            let has_tool_use = blocks
                .iter()
                .any(|b| b.get("type").and_then(|x| x.as_str()) == Some("tool_use"));
            if has_thinking && has_tool_use {
                out.events.push(LlmEvent::AssistantState(serde_json::json!({
                    "anthropic_content": blocks,
                })));
            }

            out.turn_done = true;
        }
        _ => {}
    }
    out
}

// ── Replay path: parse the genuine turn from the CLI's stdout ───────────────

/// Fold a usage snapshot into the accumulator. Anthropic splits usage across
/// `message_start` (input + cache counts) and `message_delta` (final output),
/// so we keep the non-zero fields from whichever event carried them rather than
/// letting the later zero-cache delta clobber the cache numbers.
fn merge_usage(acc: &mut UsageStats, u: UsageStats) {
    if u.prompt_tokens > 0 {
        acc.prompt_tokens = u.prompt_tokens;
    }
    if u.completion_tokens > 0 {
        acc.completion_tokens = u.completion_tokens;
    }
    if u.cache_read_tokens > 0 {
        acc.cache_read_tokens = u.cache_read_tokens;
    }
    if u.cache_creation_tokens > 0 {
        acc.cache_creation_tokens = u.cache_creation_tokens;
    }
    acc.total_tokens = acc.prompt_tokens + acc.completion_tokens;
}

/// Strip the `mcp__agentix__` namespace the CLI adds to MCP tool names so the
/// caller sees the bare tool names it registered.
fn strip_event_prefix(ev: LlmEvent) -> LlmEvent {
    match ev {
        LlmEvent::ToolCallChunk(mut c) => {
            c.name = strip_mcp_prefix(&c.name);
            LlmEvent::ToolCallChunk(c)
        }
        other => other,
    }
}

/// Drive the replay turn by parsing the CLI's own stdout. The CLI emits one
/// `stream_event` per Anthropic SSE event (already decompressed by the CLI);
/// the first `skip_turns` turns are the replayed history and the next turn is
/// the genuine passthrough generation — we parse that one for output + real
/// cache usage, without ever intercepting the upstream's bytes or touching the
/// request.
fn proxy_event_stream(
    guard: Cleanup,
    mut child: Child,
    skip_turns: usize,
) -> BoxStream<'static, LlmEvent> {
    use crate::raw::anthropic::response::StreamEvent as AStreamEvent;
    use crate::raw::anthropic::{BlockBuild, assistant_state_from_blocks, finalize, parse_stream_event};
    use crate::types::StreamBufs;

    stream! {
        let guard = guard;
        let Some(out) = child.stdout.take() else {
            yield LlmEvent::Error("claude stdout unavailable".into());
            return;
        };
        let mut lines = BufReader::new(out).lines();

        let mut bufs = StreamBufs::new();
        let mut blocks: Vec<Option<BlockBuild>> = Vec::new();
        let mut acc_usage = UsageStats::default();
        let mut usage_seen = false;
        let mut done_turns = 0usize;
        let mut saw_stop = false;

        loop {
            tokio::select! {
                line = lines.next_line() => {
                    let Ok(Some(line)) = line else { break };
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
                    if v.get("type").and_then(|t| t.as_str()) != Some("stream_event") {
                        continue;
                    }
                    let Some(event) = v.get("event") else { continue };
                    let Ok(ev) = serde_json::from_value::<AStreamEvent>(event.clone()) else {
                        continue;
                    };
                    let is_stop = matches!(ev, AStreamEvent::MessageStop);

                    if done_turns < skip_turns {
                        // Still replaying faked history; only track turn boundaries.
                        if is_stop { done_turns += 1; }
                        continue;
                    }

                    // The genuine passthrough turn.
                    if is_stop { saw_stop = true; }
                    for lev in parse_stream_event(ev, &mut bufs, &mut blocks) {
                        match lev {
                            LlmEvent::Usage(u) => { usage_seen = true; merge_usage(&mut acc_usage, u); }
                            other => yield strip_event_prefix(other),
                        }
                    }
                    if saw_stop { break; }
                }
                status = child.wait() => {
                    if !saw_stop {
                        match status {
                            Ok(s) => yield LlmEvent::Error(format!("claude exited before real turn completed ({s})")),
                            Err(e) => yield LlmEvent::Error(format!("wait claude: {e}")),
                        }
                    }
                    break;
                }
            }
        }

        if saw_stop {
            for tc in finalize(&mut bufs) {
                yield LlmEvent::ToolCall(ToolCall { name: strip_mcp_prefix(&tc.name), ..tc });
            }
            if usage_seen {
                yield LlmEvent::Usage(acc_usage);
            }
            if let Some(state) = assistant_state_from_blocks(&blocks) {
                yield LlmEvent::AssistantState(state);
            }
            yield LlmEvent::Done;
        }

        drop(child);
        drop(guard);
    }
    .boxed()
}

// ── stream_claude_code ──────────────────────────────────────────────────────

pub(crate) async fn stream_claude_code(
    _token: &str,
    _http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    let Started {
        guard,
        mut child,
        skip_turns,
    } = start_claude(config, messages, tools).await?;

    // Replay (tool-loop) path: the genuine turn is on the CLI's own stdout
    // after the faked replay turns — parse that, no wire interception.
    if let Some(skip) = skip_turns {
        return Ok(proxy_event_stream(guard, child, skip).boxed());
    }

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ApiError::Other("claude subprocess has no stdout".into()))?;
    let mut lines = BufReader::new(stdout).lines();

    let cc_debug = std::env::var("AGENTIX_CC_DEBUG").is_ok();
    Ok(stream! {
        // Moved into the generator; explicit drops below order cleanup.
        let guard = guard;
        let mut child = child;
        let mut state = StreamState::default();
        let mut got_terminal = false;

        'outer: loop {
            let line = match lines.next_line().await {
                Ok(Some(l)) => l,
                Ok(None) => break,
                Err(e) => {
                    yield LlmEvent::Error(format!("read stdout: {e}"));
                    got_terminal = true;
                    break;
                }
            };
            if line.trim().is_empty() { continue; }
            if cc_debug {
                eprintln!("[cc-stdout] {line}");
            }
            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, line = %line, "malformed stream-json line");
                    continue;
                }
            };

            // Translate stream-json event into LlmEvents. `turn_done` flips
            // when we see `message_delta` (claude's true end-of-turn signal):
            // tool_use blocks have already been flushed via `content_block_stop`,
            // and final usage rides on the `message_delta` itself.
            let outcome = translate_stream_event_line(&v, &mut state);
            for ev in outcome.events {
                yield ev;
            }
            if outcome.turn_done {
                yield LlmEvent::Done;
                got_terminal = true;
                break 'outer;
            }

            // `result` only fires after claude-code's *internal* multi-turn
            // loop ends; with stream-json input we typically break on
            // `message_delta` long before that. Keep this as a safety net for
            // turns that finish without a `message_delta` (e.g. errors).
            let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
            if ty == "result" {
                let subtype = v.get("subtype").and_then(|x| x.as_str()).unwrap_or("");
                let is_error = v.get("is_error").and_then(|x| x.as_bool()).unwrap_or(false);
                if subtype == "success" && !is_error {
                    yield LlmEvent::Done;
                } else {
                    warn!(payload = %v, "claude-code non-success result");
                    let msg = v.get("result")
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            if subtype.is_empty() {
                                "unknown error".to_string()
                            } else {
                                subtype.to_string()
                            }
                        });
                    yield LlmEvent::Error(msg);
                }
                got_terminal = true;
                break 'outer;
            }
        }

        if !got_terminal {
            match child.wait().await {
                Ok(status) if status.success() => {
                    yield LlmEvent::Error(
                        "claude exited without emitting message_delta or result".into(),
                    );
                }
                Ok(status) => {
                    yield LlmEvent::Error(format!("claude exited with status {status}"));
                }
                Err(e) => {
                    yield LlmEvent::Error(format!("wait claude: {e}"));
                }
            }
        }

        drop(child);
        drop(guard);
    }
    .boxed())
}

// ── complete_claude_code ────────────────────────────────────────────────────

pub(crate) async fn complete_claude_code(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Result<CompleteResponse, ApiError> {
    // Aggregate the streaming events into a single `CompleteResponse`. Sharing
    // the parser keeps the `assistant`-checkpoint vs `message_delta` boundary
    // logic in exactly one place.
    let mut stream = stream_claude_code(token, http, config, messages, tools).await?;

    let mut content_buf = String::new();
    let mut reasoning_buf = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = UsageStats::default();

    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => content_buf.push_str(&t),
            LlmEvent::Reasoning(t) => reasoning_buf.push_str(&t),
            LlmEvent::ToolCall(tc) => tool_calls.push(tc),
            LlmEvent::Usage(u) => usage = u,
            LlmEvent::Error(e) => return Err(ApiError::Llm(e)),
            LlmEvent::Done => break,
            _ => {}
        }
    }

    let finish_reason = if tool_calls.is_empty() {
        FinishReason::Stop
    } else {
        FinishReason::ToolCalls
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
        provider_data: None,
        usage,
        finish_reason,
    })
}
