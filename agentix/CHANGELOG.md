## [Unreleased]

### New features

- **`server-openai-responses` feature** — OpenAI Responses API-compatible HTTP server. `POST /v1/responses` (streaming SSE + non-streaming JSON). Required for clients that have deprecated Chat Completions support — notably **Codex CLI**, which only speaks Responses now. Full feature surface: input as string OR typed item array, instructions, function tools (flat shape), tool_choice (named or `{type:"function",name}`), reasoning config, multi-turn via `previous_response_id`, encrypted reasoning round-trip via `provider_data`. The streaming state machine emits the full Responses event set: `response.created` / `in_progress`, `output_item.added`/`done`, `content_part.added`/`done`, `output_text.delta`/`done`, `function_call_arguments.delta`/`done`, `reasoning_text.delta`/`done`, `response.completed` (with cumulative `output[]` snapshot).
- **In-memory session store** for `previous_response_id` chaining. Bounded LRU + 1-hour TTL by default. Each successful response (streaming or non-streaming) persists the resolved input + output items keyed by response_id, so the next turn's `previous_response_id` walks the chain to recover history. Codex defaults `store=true` and relies on this.
- **`server-openai-chat` feature** — OpenAI Chat Completions-compatible HTTP server. `POST /v1/chat/completions` (streaming SSE chunks + non-streaming JSON), `GET /v1/models`. Reasoning is exposed as `reasoning_content` on both inbound assistant messages and outbound delta/non-streaming responses (matches DeepSeek/Kimi/etc. convention).
- The `agentix` CLI now serves both formats simultaneously on the same bind port — `/v1/messages` for Anthropic clients (Claude Code, claude-code-router) and `/v1/chat/completions` for OpenAI-compat clients (Cline, Continue, openai-python, vLLM clients). Selection is implicit by request path; the upstream fallback chain is shared.
- Inbound translator handles the Chat Completions specifics that diverge from Anthropic: multi-`system`-message concatenation, `tool_call_id` ↔ agentix `Message::ToolResult { call_id }` mapping, `data:image/...;base64,...` URL parsing, `max_completion_tokens` precedence over `max_tokens`, named/object `tool_choice`, OpenAI-shape error envelope (`{"error":{"message","type","param","code"}}`).
- Streaming chunk state machine maps agentix's `LlmEvent`s to `chat.completion.chunk` deltas: text → `delta.content`, reasoning → `delta.reasoning_content`, tool calls → `delta.tool_calls[]` with stable per-tool slot index (only the first chunk for each slot includes `id`/`type`/`name`; subsequent chunks are arguments-only). `stream_options.include_usage` triggers a trailing usage chunk; the stream terminates with `data: [DONE]`. Anthropic-only `LlmEvent::ReasoningSignature` is silently dropped (no Chat Completions field for it).

### Internal changes

- Server module reorganized: `server/translated.rs` (`Translated` intermediate, format-agnostic) and `server/fallback.rs` (`UpstreamSpec`, `complete_with_fallback`, `stream_with_fallback`) are now shared across all readers. `anthropic/{inbound,fallback}.rs` previously held these.
- `AnthropicServer::router()` no longer mounts `/v1/models` when the `server-openai-chat` feature is enabled — the OpenAI Chat router owns that path under the merged binding (Anthropic clients tolerate 404 for `/v1/models`).

### Tests

- 23 new openai_responses unit tests (10 inbound, 6 outbound state machine, 4 session store, 3 misc).
- 16 openai_chat unit tests (10 inbound, 6 chunk state machine).
- E2E verified against the official `openai` Python SDK against a `claude-code` upstream:
  - **Chat Completions** (`/v1/chat/completions`) — non-streaming, streaming with `include_usage`, forced function tool calls.
  - **Responses API** (`/v1/responses`) — non-streaming, **multi-turn via `previous_response_id`** (validates session store), streaming with `response.output_text.delta` events, forced function tool calls.

### Public API

Additive only. New: `agentix::server::OpenAIChatServer` (gated by `server-openai-chat`); `agentix::server::OpenAIResponsesServer` (gated by `server-openai-responses`); `agentix::server::Translated` re-exported from the new `server::translated` module; `agentix::server::UpstreamSpec` re-exported from the new `server::fallback` module (was `server::anthropic::fallback::UpstreamSpec` — both paths still work via the existing re-export at `server::anthropic::UpstreamSpec`).

---

## [0.24.0] - 2026-05-09

### Summary

agentix flips from a pure client library into a **client + protocol-translation hub**. New `server-anthropic` feature exposes any agentix upstream chain as an Anthropic Messages-compatible HTTP endpoint, so tools that hardcode Anthropic's wire shape (Claude Code, claude-code-router, etc.) can talk to *any* agentix-supported backend. Bundled `agentix` CLI binary makes it a one-liner. Headline use case: route Claude Code at the local `claude` OAuth subscription with a paid-API fallback; verified end-to-end against the official Anthropic Python SDK.

### New features

- **`server-anthropic` feature** — Anthropic Messages-compatible HTTP server. `POST /v1/messages` (streaming SSE + non-streaming JSON), `POST /v1/messages/count_tokens`, and `GET /v1/models` are exposed. Inbound requests translate to agentix's internal `Request` + `Vec<Message>` representation; outbound, the existing 10 providers serve as backends.
- **Fallback chain** — `AnthropicServer::new(Vec<UpstreamSpec>)` accepts an ordered list of upstreams. Errors before a stream commits to its first event fall through to the next upstream; errors after commit propagate to the client. Default per-upstream pre-commit timeout: 30 s. Implementation uses peek-then-commit: the first event is awaited before any bytes are written to the client, so swapping upstreams stays transparent.
- **`agentix` CLI** — new binary behind the `cli` feature flag. Hand-rolled argv walker; repeated `-i <upstream>` declarations form the fallback chain, and trailing `--token / --model / --base-url` flags bind to the most recent `-i`. URLs are auto-recognized and routed via `Provider::OpenRouter` with the trailing `/chat/completions` stripped.
  ```
  agentix -i claude-code \
          -i https://api.deepseek.com/chat/completions --token $DEEPSEEK_API_KEY \
          --listen 127.0.0.1:7878
  ```
- **`LlmEvent::ReasoningSignature(String)`** — new variant emitted inline by `Provider::Anthropic` when the upstream sends a `signature_delta`. Enables end-to-end signature passthrough through the proxy (so `client → agentix → Anthropic → agentix → client` round-trips preserve thinking-block signatures, which Anthropic enforces on the next turn when both `thinking` and `tool_use` are present). Other providers don't emit this variant.

### Internal changes

- `raw/anthropic/{request,response}.rs` types now derive both `Serialize` and `Deserialize` (previously request types were Serialize-only and response types were Deserialize-only). `SystemBlock.kind` and `RequestMessage.role` change from `&'static str` to `String` for symmetric trait support — wire output is identical, but this is a public type-signature change for any downstream code that constructs these structs directly.
- The server's inbound translator preserves the FULL assistant block array (including signatures) into `provider_data["anthropic_content"]` whenever any thinking block is present, bypassing the existing `complete_anthropic` gate (which only saved provider_data on `thinking + tool_use` turns). Pure-thinking turns now round-trip cleanly through the proxy.

### Tests

- 23 new tests: 6 CLI-parser unit tests, 7 inbound-translation unit tests, 5 SSE state-machine unit tests, 5 server integration tests.
- E2E verified against the official `anthropic` Python SDK against a `claude-code` upstream — non-streaming and streaming both produce SDK-parseable wire output (no pydantic validation errors, signatures and usage round-trip correctly).

### Public API

- Additive: `agentix::server::{AnthropicServer, UpstreamSpec, ServerError}` (gated by `server-anthropic`); `LlmEvent::ReasoningSignature` variant (`#[non_exhaustive]` keeps it non-breaking for external matches with `_` arms).
- Breaking only for direct constructors of `raw::anthropic::request::{SystemBlock, RequestMessage}` — those fields move from `&'static str` to `String`. Internal users via the `Request` builder are unaffected.

---

## [0.23.0] - 2026-05-08

### Summary

Two behaviour fixes in providers that were previously silently producing the wrong wire format:

- **`Provider::ClaudeCode`** now correctly handles multi-turn tool loops. The provider was treating intermediate `assistant` snapshot events as turn-final, killing the subprocess before the model could emit `tool_use` blocks (typical pattern when extended thinking runs first). The end-of-turn signal is now `stream_event { type: "message_delta" }`, which is what the wire actually means. Final `usage` (including `output_tokens`) is also taken from `message_delta` instead of an early `assistant` snapshot.
- **`Provider::Mimo`** is split out of the shared Anthropic implementation (`raw::mimo::*`) and brought into line with [Xiaomi MiMo's documented spec](https://api.xiaomimimo.com): `thinking.type` only emits `enabled`/`disabled` (never the Anthropic-only `adaptive`); `output_config` is no longer sent; `max_tokens` is optional and omitted when not set, letting Mimo apply its per-model defaults; `cache_control` is dropped (Mimo's server-side cache works without client tags); `repetition_truncation` stop_reason maps to `FinishReason::Length`.

### New features

- **Claude Code reasoning round-trip** — when an assistant turn contains both thinking and tool_use blocks, the raw blocks (incl. `signature`) are now captured into `LlmEvent::AssistantState` / `Message::Assistant.provider_data` under the same `anthropic_content` envelope used by the Anthropic provider. On the next turn, `assistant_replay_message` re-emits them verbatim on stdin, preserving the model's chain-of-thought across `claude -p` invocations and avoiding redundant re-thinking.

### Fixes

- `Provider::ClaudeCode`: extended-thinking turns are no longer truncated before `tool_use` is emitted (root cause of agent loops terminating after one turn).
- `Provider::ClaudeCode`: per-turn `completion_tokens` now reflects the final value from `message_delta`, not an early snapshot.
- `Provider::Mimo`: removes invalid `thinking: {type: "adaptive"}` and `output_config` fields that the spec doesn't accept.

### Public API

No breaking changes to the Rust API. `Provider::Mimo` and `Request::mimo()` keep their signatures; only the wire body emitted to Mimo changes.

### Tests

- 11 new `raw::mimo` unit tests covering the spec-divergent fields.
- New `tests/smoke_mimo.rs` (`#[ignore]`) live integration: complete / stream+thinking / multi-turn tool loop. Run with `MIMO_API_KEY=… MIMO_BASE_URL=… cargo test --test smoke_mimo -- --ignored`.

---

## [0.22.0] - 2026-04-25

### Summary

Document (PDF etc.) support added to the multimodal `Content` enum. Providers that natively support files (Anthropic, OpenAI Responses, Gemini, OpenRouter) emit them on the wire; OpenAI-chat-compat providers that don't (DeepSeek, Grok, GLM, Kimi) silently drop them.

### New features

- `Content::Document(DocumentContent)` variant with `DocumentData::{Base64, Url}`. Optional `filename` (OpenAI's `input_file` requires one alongside `file_data`; a default is supplied if absent).
- **Anthropic** — emits `{"type": "document", "source": {"type": "base64" | "url", ...}}` blocks. `cache_control` stamps apply to documents like any other block. Tool-result documents are dropped (not accepted by the API).
- **OpenAI Responses** — emits `{"type": "input_file", "file_data" | "file_url", "filename"}` content parts.
- **Gemini** — base64 routes to `inline_data`, URLs route to `file_data` (`{mime_type, file_uri}`).
- **OpenRouter** — emits `{"type": "file", "file": {"filename", "file_data"}}` (the PDF plugin shape).
- **claude-code session** — mirrors the Anthropic document block shape on stdin JSON.

### Non-supporting providers

DeepSeek, Grok, GLM, Kimi silently drop `Content::Document` parts (both in user messages and tool results). Users who need PDFs with these models route through `Provider::OpenRouter` with a matching `base_url`.

### Breaking changes

- `Content` enum gains a new variant. Exhaustive external `match` on it must add a `Content::Document(_)` arm.

### Fixes

- None.

---

## [0.21.0] - 2026-04-25

### Summary

Opaque reasoning round-trip now wired end-to-end for **OpenAI** (switched to Responses API), **Gemini** (`thoughtSignature`), and **OpenRouter** (`reasoning_details[]`). All three plug into the `provider_data` / `LlmEvent::AssistantState` plumbing added in 0.20.0 — no public API changes for consumers who just stream tokens and read tool calls.

### New features

**OpenAI — Responses API (`/v1/responses`)**
- `Provider::OpenAI` now targets the Responses API exclusively. Stateless mode (`store: false`, `include: ["reasoning.encrypted_content"]`) — never uses `previous_response_id`.
- `LlmEvent::Reasoning(text)` streams OpenAI's reasoning summary text (previously invisible on Chat Completions). Full CoT stays encrypted in `encrypted_content`; round-trips via `provider_data` (envelope tag `openai_responses_items`).
- `reasoning_effort` maps to the `reasoning: {effort}` request config (`minimal/low/medium/high/xhigh`).
- System prompt → top-level `instructions`. Structured output → `text.format`. Tools use the flat function shape. Tool choice uses the flat `{type: "function", name}` shape.
- `UsageStats.reasoning_tokens` populated from `output_tokens_details.reasoning_tokens`.
- Unknown output-item and SSE event types are tolerated via `#[serde(other)]` — forward-compatible with future Responses API additions.

**Gemini — thinking config + `thoughtSignature` round-trip**
- `generationConfig.thinkingConfig` wired up. Model-sniffed:
  - `gemini-3*` → `thinkingLevel` (`minimal/low/medium/high`)
  - `gemini-2.5*` → `thinkingBudget` (integer tokens, mapped from the enum)
  - Other models — no thinking config emitted
- `ResponsePart` now captures `thought: bool` and `thoughtSignature` fields. Streaming `LlmEvent::Reasoning` fires for thought parts.
- Multi-turn round-trip: the raw `content.parts[]` array is captured into `provider_data` (envelope tag `gemini_parts`). Gemini 3 enforces `thoughtSignature` presence on the first `functionCall` part per step or 400s; agentix now always round-trips the full parts array to satisfy this.
- `UsageStats.reasoning_tokens` populated from `usageMetadata.thoughtsTokenCount`.

**OpenRouter — unified `reasoning` param + typed `reasoning_details[]` round-trip**
- Outgoing request uses the unified `reasoning: {effort: ...}` body field (normalizes across underlying providers) instead of raw `reasoning_effort` passthrough.
- Response parsing now captures `reasoning_details[]` (`reasoning.text` / `reasoning.summary` / `reasoning.encrypted` entry types). Plaintext reasoning still surfaces via `LlmEvent::Reasoning`.
- Streaming reassembles fragmented `reasoning_details` entries by their `index` field (guards against LangChain #36400-style fragmentation producing malformed multi-turn requests).
- Round-trip via `provider_data` (envelope tag `openrouter_reasoning_details`). Preserves `signature` / encrypted `data` fields the underlying provider needs.

**Cross-provider: `UsageStats.reasoning_tokens`**
- New `usize` field on `UsageStats`, populated by OpenAI, Gemini (and any future provider that reports it). `AddAssign` accumulates it across turns.

### Breaking changes

- **`Provider::OpenAI` no longer speaks Chat Completions.** Any OpenAI-compatible server without `/v1/responses` support (Azure legacy deployments, vLLM, LocalAI, Ollama, llama.cpp server, etc.) must migrate to `Provider::OpenRouter` with a custom `base_url`. Other in-tree providers (Grok, Kimi, GLM, MiniMax, OpenRouter, DeepSeek) are unaffected — they each speak their own Chat-Completions-compatible endpoint.
- **`UsageStats` gains `reasoning_tokens: usize`**. Additive with default 0 — `..Default::default()` constructions are unaffected; exhaustive struct-literal constructions break until updated.
- **OpenAI `extra_body` knobs for `frequency_penalty`/`presence_penalty`/`logit_bias`** now hit an API error (Responses API rejects them). Users needing them switch to `Provider::OpenRouter`.
- **`raw::openai::*`** type surface rewritten for Responses API. Directly importing raw types from that module is breaking; consumers should use the public `Request`/`Message` API instead.

### Test fixtures

- Chat-Completions-shaped fixtures moved from `tests/fixtures/openai/` to `tests/fixtures/chat_completions/` (DeepSeek uses them).
- New Responses-API-shaped fixtures in `tests/fixtures/openai/`.
- New `tests/fixtures/gemini/complete_thinking_tool_call.json` exercising `thoughtSignature` round-trip.

---

## [0.20.0] - 2026-04-25

### Summary

Cross-provider reasoning control via a single `ReasoningEffort` enum, opaque per-turn provider state for round-tripping Anthropic thinking signatures, and DeepSeek/Anthropic adaptive-thinking wire support.

### New features

**`ReasoningEffort` — cross-provider reasoning hint**
- New `agentix::ReasoningEffort` enum: `None | Minimal | Low | Medium | High | XHigh | Max`.
- `None` explicitly disables thinking on providers that support a toggle (DeepSeek, Anthropic).
- Any other value engages thinking; `Request::reasoning_effort(e)` is the single builder entry point.
- Providers coerce to their own scale automatically — see README for the mapping table.

**`Message::Assistant.provider_data` — opaque per-turn state**
- New `provider_data: Option<serde_json::Value>` field carries provider-specific state (currently: Anthropic thinking-block sequences with signatures) across turns.
- Only the producing provider's raw layer consumes it; other providers ignore it transparently.

**`LlmEvent::AssistantState` + `#[non_exhaustive]`**
- New variant carries opaque per-turn state from the stream; the agent loop attaches it to the reconstructed `Message::Assistant.provider_data`.
- `LlmEvent` is now `#[non_exhaustive]`, so future variants will not break downstream match arms (add `_ => {}` to your matchers).

**DeepSeek: thinking mode derived from effort**
- `reasoning_effort(None)` → `{"thinking": {"type": "disabled"}}` + sampling params flow through.
- Any other effort → `{"thinking": {"type": "enabled"}}` + `reasoning_effort: "high" | "max"` (Minimal/Low/Medium/High collapse to `high`, XHigh/Max collapse to `max`).
- Unset effort → `{"thinking": {"type": "enabled"}}` with no effort (DeepSeek's own default, currently `high`).
- Sampling-only parameters (`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`) are type-level incompatible with thinking mode — setting them while thinking is on drops them before the wire with a tracing warn, matching the API's "accepted but ignored" semantics.

**Anthropic: adaptive thinking + signature round-trip**
- `build_anthropic_request` emits `thinking: {type: "adaptive" | "disabled"}` and `output_config: {effort: ...}` derived from `ReasoningEffort`.
- Response parser preserves `signature` on thinking blocks, adds `redacted_thinking` support, and consumes `signature_delta` in streams.
- When a turn contains both thinking **and** tool_use blocks, the raw layer captures the full content array (thinking + tool_use + text, interleaved) into `CompleteResponse.provider_data` / `LlmEvent::AssistantState`. On the next turn, the request serializer emits these blocks verbatim to preserve the signature ordering Anthropic enforces.

### Breaking changes

- **`Message::Assistant` has a new required field `provider_data: Option<serde_json::Value>`.** Any code constructing it directly must add `provider_data: None`.
- **`CompleteResponse` has a new required field `provider_data: Option<serde_json::Value>`.** Any code constructing it directly must add `provider_data: None`.
- **`LlmEvent` is now `#[non_exhaustive]` and has a new variant `AssistantState(Value)`.** Existing exhaustive matchers in downstream crates should add `_ => {}` (or explicitly handle `AssistantState`).
- **No more `Request::thinking(bool)` builder.** Use `reasoning_effort(ReasoningEffort::None)` to explicitly disable thinking, or any other `ReasoningEffort` value to engage it. Leaving `reasoning_effort` unset keeps the provider default.

---

## [0.13.0] - 2026-04-11

### New features

**`FinishReason` — why the model stopped**
- New `agentix::FinishReason` enum: `Stop`, `Length`, `ToolCalls`, `ContentFilter`, `Other(String)`.
- `CompleteResponse` now has a `finish_reason: FinishReason` field (non-optional, defaults to `Stop`).
- `FinishReason::is_truncated()` convenience method.
- Mapped from all three providers: OpenAI/Gemini `finish_reason`, Anthropic `stop_reason`.

### Breaking changes

- `CompleteResponse` has a new required field `finish_reason: FinishReason`. Any code that constructs `CompleteResponse { ... }` directly will need to add `finish_reason: FinishReason::Stop` (or the appropriate variant).

---

## [0.12.0] - 2026-04-11

### Summary

Multimodal tool results, Gemini tool-calling fixes, and a unified `Content` type across the message layer.

### New features

**`Content` — unified content block type**
- New `agentix::Content` enum (`Text { text }` / `Image(ImageContent)`) replaces the former `UserContent` alias.
- `pub type UserContent = Content` is retained for backwards compatibility.
- `Content::text(s)` convenience constructor.

**Multimodal tool results**
- `Message::ToolResult.content` is now `Vec<Content>` instead of `String`, enabling tools to return images alongside text.
- `AgentEvent::ToolResult.content` is now `Vec<Content>`.
- `AgentEvent::text()` helper — returns text parts of a `ToolResult` event joined by newlines; returns `None` for other variants.
- `ToolOutput::Result` is now `Vec<Content>` instead of `serde_json::Value`.

**Richer `#[tool]` return types**
- New `ToolResultContent` trait (highest priority in the dtolnay autoref chain) handles `Vec<Content>`, `ImageContent`, `String`, and `&str` directly without JSON serialisation.
- `String` / `&str` returning tools now produce a plain-text `Content::Text` part instead of a quoted JSON string.
- `T: Serialize` (catch-all) and `Result<T, E>` continue to work, serialising via `serde_json::to_string`.

**`ContentWire` / `content_to_wire()` in `raw::shared`**
- Internal helper used by all provider adapters: single-text results are sent as a plain string; multi-part or image results are sent as an array.

### Bug fixes

**Gemini function-calling schema**
- `FunctionDeclaration.parameters` is now passed through `sanitize_schema_for_gemini()`, which fixes `400 INVALID_ARGUMENT` errors caused by `type: [...]` arrays and `items: true` in schemars-generated schemas.

**Gemini `FunctionResponse.response` must be a JSON object**
- Tool results are now wrapped in `{ "result": ... }` when not already an object, satisfying Gemini's proto `Struct` requirement.

### Breaking changes

- `Message::ToolResult { content }` — type changed from `String` to `Vec<Content>`. Construct with `vec![Content::text("...")]`.
- `AgentEvent::ToolResult { content }` — type changed from `String` to `Vec<Content>`. Use `.text()` for the previous string behaviour.
- `ToolOutput::Result` — inner type changed from `serde_json::Value` to `Vec<Content>`.
- `agentix-macros` bumped to `0.5.0`.

---

## [0.3.0] - 2026-03-23

### Summary

Graph architecture improvements: typed edges, shared state, and lifecycle safety.

### New features

**`Graph::edge_map()` — caller-controlled edge transform**
- New method `Graph::edge_map(&from, &to, fn(Msg) -> Option<Msg>)` that lets the caller choose exactly which messages cross the edge and how they are transformed.
- Replaces the hardcoded `Token→User` conversion of `edge()` when typed payloads (e.g. `Msg::Custom`) need to be forwarded without string conversion.
- `edge()` now delegates to `edge_map()` internally.

**`GraphHandle` — explicit graph lifecycle**
- `Graph::into_handle()` consumes the graph and returns a `GraphHandle`, a pure RAII guard.
- Dropping `GraphHandle` aborts all edge background tasks — no more zombie tasks.
- No public methods: abort early via `drop(handle)`.

**`SharedContext` — global shared state for graph nodes**
- New `agentix::SharedContext` type: a `Clone`-cheap `Arc<RwLock<HashMap<String, serde_json::Value>>>` injectable into any node.
- API: `ctx.set(key, value)`, `ctx.get::<T>(key)`, `ctx.get_str(key)`, `ctx.remove(key)`, `ctx.snapshot()`.
- `PromptTemplate::context(ctx)` — attaches a `SharedContext`; unresolved `{key}` placeholders (not covered by `.var()`) are looked up in the context at render time, enabling hot-swappable template variables.

**`mcp-server` feature** (from previous unreleased)
- Expose any `ToolBundle` as a standards-compliant MCP server using the `rmcp` library.
  - `McpServer::serve_stdio()` — stdio transport for Claude Desktop / MCP Studio integration.
  - `McpServer::serve_http(addr)` — Streamable HTTP transport, mounts the MCP endpoint at `/mcp`.
  - `McpServer::into_http_service(config)` — Tower-compatible `StreamableHttpService`.
  - Builder methods `with_name()` and `with_version()`.

### Bug fixes

**`Graph::edge()` now forwards `Msg::User` messages**
- Previously, `Msg::User` emitted by `PromptTemplate` or `OutputParser` was silently dropped by the default edge transform (`_ => None`), causing the downstream node to never receive the message and hanging the pipeline.
- Fixed: the default transform now passes `Msg::User(_)` through unchanged, in addition to converting `Token(text)` → `User(text)` and passing `Custom(_)` unchanged.

### Breaking changes

- `Graph::edge_count()` previously tracked a separate `usize` counter; it now returns `self.handles.len()`. Behavior is identical.
- Old code that calls `Graph::new()...edge(...)` without `into_handle()` continues to work (dropping `Graph` does **not** abort tasks — only `GraphHandle::drop` does). However, **call `into_handle()` and keep the result alive** to ensure tasks are properly cancelled on shutdown.

---

## [0.10.2] - 2026-03-16

### Bug fixes

- Fix truncated tool output handling when forwarding results (fixes cases where long tool outputs were clipped unexpectedly). This release is a small patch to ensure full tool output is preserved or correctly truncated according to configured limits.

---

## [0.10.1] - 2026-03-16

### New features

- `McpTool` output length limits — added configurable output-size caps and safer truncation logic for MCP-backed tools so extremely large tool outputs no longer cause downstream problems or UI truncation surprises.

### Maintenance

- Bumped crate version to `0.10.1`.

---

## [0.10.0] - 2026-03-14

### Summary

Minor/feature release that improves tool handling, streaming robustness, and runtime flexibility around tools and interrupts.

### New features

- Runtime tool injection for agents — it's now possible to add (and in some cases configure) tools at runtime without rebuilding the agent. This enables dynamic tool wiring for long-running processes or interactive applications.
- `ToolCallChunk` and streaming events now include an index field to make chunk ordering explicit and more robust for reconstructing streaming tool argument payloads on the client/UI side.

### Behavior / robustness

- Interrupt and tool channels are now "always-on" — the agent drains and accepts interrupt messages and tool events deterministically during streaming and non-streaming paths, reducing lost messages and race conditions.
- Buffer interrupts during tool execution so injected messages arriving while a tool is running are reliably queued and applied between tool rounds.
- Preserve all events per chunk using an internal pending-events queue to avoid losing any intermediate tool-call or reasoning events when event handling is back-pressured.
- Tool method bodies wrapped in an explicit async closure to avoid `return` accidentally escaping outer `call` functions — improves safety of tool implementations and avoids surprising early returns.

### Notes

- These changes improve streaming determinism and make UIs simpler to implement (clients can rely on per-chunk indices and stable event ordering).

---

## [0.8.3] - 2026-03-12

### New features

- `ToolBundle` — a convenience grouping for registering multiple tools together with a single call. Useful for packaging related tool sets and keeping agent setup code tidy.

### Maintenance / quality-of-life

- Ignore common macOS artifact files (`.DS_Store`) across tooling and CI paths.
- Insert the configured system prompt at the top of assembled requests to ensure provider-side behavior is consistent with agent-level system prompts.

---

## [0.8.2] - 2026-03-12

### New features

- Added `extra_body` support to the request types: `ChatCompletionRequest` now includes an `extra_body: Option<Map<String, Value>>` field that is annotated with `#[serde(flatten)]`. When present, these key/value pairs are merged into the top-level JSON request body so callers can pass provider-specific or experimental fields not yet modelled by the typed request.

- Added fine-grained helper APIs for adding single fields:
  - `ChatCompletionRequest::add_extra_field(&mut self, key, value)` — in-place insertion.
  - `ChatCompletionRequest::with_extra_field(self, key, value) -> Self` — builder-style chaining.
  - `ApiRequest::add_extra_field(&mut self, key, value) -> &mut Self` — in-place builder mutation.
  - `ApiRequest::with_extra_field(self, key, value) -> Self` — consuming builder-style helper.
  - `ApiRequest::extra_field(...)` preserved as a compatibility alias.

- `DeepseekAgent` builder helpers:
  - `DeepseekAgent::extra_body(map)` and `DeepseekAgent::extra_field(key, value)` — store extra top-level fields on the agent which are merged into the `ApiRequest` produced by `build_request`. This provides a convenient way to attach provider-specific options at the agent level.

### Tests

- Added unit test `test_extra_body_serialize_merge` verifying that `extra_body` entries are flattened into the top-level JSON when a `ChatCompletionRequest` is serialized.

### Documentation

- README updated with a new "Custom top-level request fields (extra_body)" section that documents usage patterns for `ApiRequest::extra_body`, `ApiRequest::with_extra_field`, and the `DeepseekAgent` helpers, including examples and notes about key collision risks.

### Notes

- `extra_body` fields are flattened into the top-level request JSON via `serde(flatten)` and therefore should avoid colliding with existing top-level field names such as `messages` or `model`.
- `DeepseekAgent`-held `extra_body` is persisted on the agent instance until changed or cleared explicitly (it is not automatically cleared after a single request). If you want a one-shot semantics, call `ApiRequest` helpers directly for per-request control.

## [0.8.0] - 2026-03-12

### Summary

Breaking release. `reasoning_content` is no longer stripped from conversation history. Previously it was erased in-place on assistant messages without `tool_calls` at the start of every new turn, permanently destroying it for persistence and display. It is now filtered on the outbound request copy only, so callers can rely on history being immutable.

### Breaking changes

**`reasoning_content` preservation semantics changed**

In `0.7.x`, calling `agent.chat(...)` or `agent.chat_from_history(...)` had the side effect of mutating conversation history: `reasoning_content` was silently stripped from any assistant message that lacked `tool_calls`. This meant that after the first new turn, `agent.history()` no longer contained any reasoning content, and any downstream serialization (e.g. writing history back to a database) would permanently lose it.

In `0.8.0`, history is never mutated for this purpose. The outbound `messages` array sent to the API is filtered instead, according to deepseek-reasoner's rules:

1. Assistant messages **with** `tool_calls` **keep** their `reasoning_content` — the model needs it to continue reasoning after seeing tool results.
2. Among those, only the **last** such message keeps its `reasoning_content`; earlier ones are stripped to reduce prompt size.
3. Assistant messages **without** `tool_calls` always have `reasoning_content` omitted from the request (sending it there causes a 400 error on the next tool-calling turn).

**Migration:** if your code reads `agent.history()` after a turn and expects `reasoning_content` to have been cleared, it will now still be present. Filter it yourself at read time if needed. No changes are required if you were simply displaying or persisting history — this release is strictly better for those use cases.

### Bug fixes

**`reasoning_content` no longer lost after the first new turn**

`DeepseekAgent::drain_interrupts` previously mutated `conversation.history_mut()` to erase `reasoning_content` from assistant messages without `tool_calls`. This permanently destroyed the data before it could be serialized or displayed. The mutation has been removed; filtering now happens in `build_request` on a cloned message list.

---

## [0.6.0] - 2026-03-10

### Summary

Breaking release. `AgentEvent` has been simplified: three separate tool-call event variants are replaced by a single `ToolCall(ToolCallChunk)` that works uniformly in both streaming and non-streaming modes — the same pattern as `Token`. Also adds `deepseek-reasoner` support via a new `ReasoningToken` event, and fixes the interrupt channel for streaming agents.

### Breaking changes

**`AgentEvent` — tool call variants unified**

`ToolCallStart`, `ToolCallArgsDelta`, and `ToolCall(ToolCallInfo)` are replaced by a single variant:

```rust
AgentEvent::ToolCall(ToolCallChunk { id, name, delta })
```

Behaviour mirrors `Token`:
- **Streaming**: one event per SSE chunk. First chunk has `delta = ""` (name is now known — create your UI element). Subsequent chunks carry incremental argument JSON in `delta`.
- **Non-streaming**: one event per tool call with the complete argument JSON in `delta`.

Accumulate `delta` values by `id` to reconstruct the full argument string. `ToolResult` marks completion.

**`ToolCallInfo` removed** — replaced by `ToolCallChunk`:

```rust
// before
AgentEvent::ToolCall(info)       // info.id, info.name, info.args: Value
AgentEvent::ToolCallStart { id, name }
AgentEvent::ToolCallArgsDelta { id, delta }

// after
AgentEvent::ToolCall(c)          // c.id, c.name, c.delta: String
```

**`ToolCallResult.args` type changed: `Value` → `String`**

The raw JSON string from the wire is now passed through directly. Parse it yourself if you need a structured object.

### Migration

Replace three match arms with one:

```rust
// before
Ok(AgentEvent::ToolCallStart { id, name }) => { /* show tool name */ }
Ok(AgentEvent::ToolCallArgsDelta { id, delta }) => { /* accumulate */ }
Ok(AgentEvent::ToolCall(info)) => { /* info.name, info.args */ }

// after
Ok(AgentEvent::ToolCall(c)) => {
    if c.delta.is_empty() {
        // first chunk: c.name is known, args not yet streaming
    } else {
        // subsequent chunks: append c.delta to your buffer
    }
}
// in non-streaming mode, c.delta holds the complete args on the single event
```

---

### New features

**`AgentEvent::ReasoningToken(String)` — deepseek-reasoner support**

A new event variant carrying incremental thinking/reasoning content from models that expose it (e.g. `deepseek-reasoner`). Arrives token-by-token before the main reply in streaming mode; arrives in full in non-streaming mode. Absent for models that do not produce reasoning content.

```rust
Ok(AgentEvent::ReasoningToken(t)) => print!("<think>{t}</think>"),
```

**`reasoning_content` round-trip for multi-turn deepseek-reasoner**

`reasoning_content` is now preserved in history on assistant messages that contain tool calls (required by deepseek-reasoner to continue reasoning after seeing tool results), and stripped from plain reply messages at the start of the next turn (sending it there causes a 400 error). Previously, both multi-turn tool-calling and reasoning content caused API 400 errors.

### Bug fixes

**Interrupt channel now works during streaming**

`ApiClient::send()` now routes through the interrupt-aware path when streaming is enabled, so injected messages are correctly queued mid-generation. Previously, interrupts sent while an SSE stream was open were silently dropped.

---

## [0.5.6] - 2026-03-10

### Summary

New `mcp` optional feature: connect any MCP server's tools to `DeepseekAgent` with a single dependency line.

### New features

**`McpTool` — MCP client support (`features = ["mcp"]`)**

- New `ds_api::McpTool` implements the `Tool` trait and can be passed directly to `DeepseekAgent::add_tool()`.
- Two transports supported:
  - `McpTool::stdio(program, args)` — spawns a child process (`npx`, `uvx`, or any binary) and communicates over stdin/stdout.
  - `McpTool::http(url)` — connects to a remote MCP server over Streamable HTTP.
- At construction time, `tools/list` is called automatically (pagination handled transparently) and the tool list is cached. Each subsequent model tool call is forwarded via `tools/call`.
- The MCP server's `inputSchema` is passed through as-is to the DeepSeek API `parameters` field — no manual schema configuration needed.
- New `ds_api::mcp::McpError` error type covering process spawn failure, handshake failure, tool list fetch failure, and tool call failure.

**Usage**

```toml
[dependencies]
ds-api = { version = "0.5", features = ["mcp"] }
```

```rust
use ds_api::{DeepseekAgent, McpTool};

let agent = DeepseekAgent::new(token)
    .add_tool(McpTool::stdio("npx", &["-y", "@playwright/mcp"]).await?)
    .add_tool(McpTool::stdio("uvx", &["mcp-server-git"]).await?)
    .add_tool(McpTool::http("https://mcp.example.com/").await?);
```

## [0.5.4] - 2026-03-09

### Summary
Bug fix: interrupt channel messages were silently dropped on turns where the model returned a plain text response with no tool calls.

### Bug fixes

**Interrupt channel now drained before every API turn**
- Previously, `with_interrupt_channel()` messages were only picked up inside `execute_tools()`, meaning any message sent during a no-tool turn was never inserted into the conversation history.
- `drain_interrupts()` is now also called at the top of the `Idle` state transition, so queued messages are always flushed before the next API call regardless of whether tools were used in the previous turn.

### Notes
- No breaking changes — all `0.5.x` code continues to compile unchanged.

---

## [0.5.3] - 2026-03-09

### Summary
Mid-loop user message injection via an interrupt channel. No breaking changes — all existing `0.5.x` code continues to compile unchanged.

### New features

**`DeepseekAgent::with_interrupt_channel()`**
- New builder method that attaches an `UnboundedSender<String>` to the agent.
- Returns `(DeepseekAgent, InterruptSender)` — the agent and the sender half of the channel.
- Any message sent through the `InterruptSender` is picked up automatically after the current tool-execution round finishes and appended to the conversation history as a `Role::User` message before the next API turn.
- The sender can be cloned freely and used from any task or callback (e.g. a Telegram bot handler) without blocking.
- If the agent is idle (not in a tool loop), messages accumulate in the channel and are drained on the next tool round.
- Agents without an interrupt channel (the default) are unaffected — no overhead.

```rust
let (agent, tx) = DeepseekAgent::new(token)
    .with_streaming()
    .add_tool(MyTool)
    .with_interrupt_channel();

// In another task — fires while the agent is executing tools:
tx.send("Actually, use Python instead.".into()).unwrap();
```

Timing — message is injected between tool round and next API turn:
```
User prompt
  → API call → ToolCall(search)
  → tool executing…  ← tx.send("change of plan") arrives here
  → ToolResult(search)
  → drain channel → push User("change of plan") into history
  → API call (model now sees the injected message)
  → Token("Sure, pivoting to…")
```

**`DeepseekAgent::history()`**
- New public read-only accessor returning `&[Message]` — the full conversation history in order.
- Includes system prompts, user turns, assistant replies, tool calls, tool results, and any auto-summary messages.
- Previously the `conversation` field was `pub(crate)` and inaccessible from application code.

```rust
for msg in agent.history() {
    println!("{:?}: {:?}", msg.role, msg.content);
}
```

**`InterruptSender` type alias**
- `ds_api::InterruptSender` is a re-export of `tokio::sync::mpsc::UnboundedSender<String>`.
- Import it directly instead of spelling out the full `tokio` path.

### New example

**`examples/interrupt.rs`**
- Demonstrates `with_interrupt_channel()` end-to-end.
- A `SlowCounter` tool counts to 5 with a 200 ms delay per step (~1 s total).
- A background task injects a follow-up message at 500 ms (mid-tool-execution).
- After the tool round finishes, the agent incorporates the injected message into its next reply.
- Shows `stream.into_agent()` recovery and `agent.history()` inspection.

Run with:
```bash
DEEPSEEK_API_KEY=sk-... cargo run --example interrupt
```

### Notes
- All tests pass.
- No breaking changes — `0.5.2` consumers require no code changes.

---

## [0.5.2] - 2026-03-09

### Summary
OpenAI-compatible provider support.

### New features

**`DeepseekAgent::custom(token, base_url, model)`**
- New constructor for pointing the agent at any OpenAI-compatible endpoint (OpenRouter, OpenAI, local Ollama, etc.).
- All three parameters are fixed at construction time; the agent is fully configured in one call.
- The default `LlmSummarizer` is automatically initialised with the same base URL and model — no manual `ApiClient` or `LlmSummarizer` wiring required.

```rust
// DeepSeek (unchanged)
let agent = DeepseekAgent::new(token);

// Any OpenAI-compatible provider
let agent = DeepseekAgent::custom(
    token,
    "https://openrouter.ai/api/v1",
    "meta-llama/llama-3.3-70b-instruct:free",
);
```

**`DeepseekAgent::with_model(model)` / `ApiRequest::with_model(model)` / `LlmSummarizer::with_model(model)`**
- New builder method on each type accepting any `impl Into<String>` model identifier.
- Removes the need to import or construct the internal `Model` enum for custom model names.

**`Model::Custom(String)` variant**
- The internal `Model` enum gained a `Custom(String)` variant with hand-written `Serialize`/`Deserialize` that passes the string through as-is.
- `Model` is not re-exported at the crate root; callers use the string-based builders above instead.

**`system_fingerprint` made optional in responses**
- `ChatCompletionResponse` and `ChatCompletionChunk` now deserialise correctly when the provider omits `system_fingerprint` (many non-DeepSeek providers do).

### Notes
- All 49 tests (22 unit, 10 integration, 17 doctest) pass.
- No breaking changes — `0.5.1` consumers require no code changes.
- The Roadmap item "OpenAI-compatible providers" is now complete.

---

## [0.5.1] - 2026-03-09

### Summary
Internal refactor: business logic extracted from the streaming state machine into a dedicated `executor` module.  No public API changes.

### Changes

**New `agent/executor.rs` module**
- Extracted all "do actual work" functions out of `stream.rs` into a new `executor.rs`:
  - `build_request` — assembles an `ApiRequest` from history + tools.
  - `run_summarize` — runs `maybe_summarize` and transfers agent ownership back.
  - `fetch_response` — non-streaming API call; appends assistant turn to history.
  - `connect_stream` — opens an SSE `BoxStream` for the current turn.
  - `execute_tools` — dispatches all pending tool calls and collects results.
  - `finalize_stream` — assembles complete `ToolCall` objects from SSE delta buffers and records the assistant turn.
  - `apply_chunk_delta` — applies one SSE chunk delta to the `StreamingData` accumulator.
  - `raw_to_tool_call_info` — converts a wire `ToolCall` to the public `ToolCallInfo` type.
- Internal accumulator types (`FetchResult`, `ToolsResult`, `PartialToolCall`, `StreamingData`) and future type aliases (`FetchFuture`, `ConnectFuture`, `ExecFuture`, `SummarizeFuture`) moved to `executor.rs`.
- `stream.rs` now contains only the `AgentStream` state machine and its `Stream` impl — no business logic, no `async fn`s.
- This separation makes it straightforward to add retries, timeouts, or parallel tool execution in the future without touching the state machine.

**`SlidingWindowSummarizer` improvements**
- Added `trigger_at(n: usize)` builder method: set the non-system message count above which summarization is triggered, independently of the `window` (retain count).  Useful when you want the window to only slide after a burst of messages rather than on every new message.
- `trigger_at` is silently clamped to `window + 1` if a value ≤ `window` is provided.
- Default behaviour is unchanged: triggers as soon as the non-system count exceeds `window`.

**Documentation**
- `AgentStream` now has a full doc comment with an example showing streaming event handling.
- `Summarizer` trait doc includes a complete custom-summarizer example (`TurnLimitSummarizer`).
- `AgentStreamState` variants have inline doc comments explaining each state's role.
- `executor.rs` functions all have doc comments explaining inputs, outputs, and side-effects.

### Notes
- All 36 existing tests (15 unit, 10 integration, 11 doctest) continue to pass.
- No breaking changes — `0.5.0` consumers require no code changes.

---

## [0.5.0] - 2026-03-08

### Summary
Breaking release: the agent event type has been redesigned from a flat struct to a proper enum.

### Breaking changes

**Agent event type**
- `AgentResponse` struct removed. Replaced by `AgentEvent` enum.
- `ToolCallEvent` struct removed. Replaced by two focused structs:
  - `ToolCallInfo` — carries `id`, `name`, `args`; yielded before execution.
  - `ToolCallResult` — carries `id`, `name`, `args`, `result`; yielded after execution.
- `AgentStream` now implements `Stream<Item = Result<AgentEvent, ApiError>>`.
- Tool call previews and results are now yielded **one per event** (previously batched in a `Vec`).
- The old `tc.result == Value::Null` idiom for distinguishing previews from results is gone; the variant itself encodes the distinction.

**Summarizer**
- `TokenBasedSummarizer` removed.
- `Summarizer::summarize` is now `async` (returns `Pin<Box<dyn Future<Output = Result<(), ApiError>>>>`). Any custom `Summarizer` implementation must be updated.
- New default summarizer is `LlmSummarizer`, which calls DeepSeek to produce a semantic summary of older turns. It requires an `ApiClient` at construction time.
- New alternative `SlidingWindowSummarizer` replaces `TokenBasedSummarizer` for cases where zero extra API calls are desired.
- Permanent `Role::System` messages set via `with_system_prompt` are now protected and never removed by any built-in summarizer.

### Migration

Replace:
```rust
// old
use ds_api::{AgentResponse, ToolCallEvent};

while let Some(event) = stream.next().await {
    let ev = event?;
    if let Some(text) = ev.content {
        print!("{text}");
    }
    for tc in ev.tool_calls {
        if tc.result.is_null() {
            println!("[calling {}({})]", tc.name, tc.args);
        } else {
            println!("[result] {}", tc.result);
        }
    }
}
```
with:
```rust
// new
use ds_api::AgentEvent;

while let Some(event) = stream.next().await {
    match event? {
        AgentEvent::Token(text)    => print!("{text}"),
        AgentEvent::ToolCall(c)    => println!("[calling {}({})]", c.name, c.args),
        AgentEvent::ToolResult(r)  => println!("[result] {}", r.result),
    }
}
```

### Notes
- The `AgentEvent::Token` variant carries assistant text in both streaming and non-streaming modes. In streaming mode each `Token` is a single SSE delta; in non-streaming mode the full response text arrives as one `Token`.
- `ToolCall` and `ToolResult` events are emitted in matching order (first call → first result).
- `LlmSummarizer` errors (e.g. a transient API failure during summarization) are swallowed silently so an ongoing conversation is never aborted by a failed summary attempt.
- `SlidingWindowSummarizer` takes a `window: usize` argument and never makes an API call.

**Architecture**
- `DeepseekConversation` renamed to `Conversation`. The `Conversation` trait has been removed — there is now a single concrete struct with all methods defined directly on it.
- `DeepseekAgent` no longer holds a redundant `client` field; the single `ApiClient` lives inside `Conversation`.
- `AgentStream` state machine simplified: the `YieldingToolCalls` and `YieldingToolResults` states now carry their own queues (`VecDeque`) instead of storing them as loose fields on the stream struct. This makes the state machine self-contained and eliminates implicit field–state coupling.
- The `[auto-summary]` magic string is now centralised as `Message::AUTO_SUMMARY_TAG`, with `Message::is_auto_summary()` and `Message::auto_summary()` helpers. Custom `Summarizer` implementations should use these instead of comparing name strings directly.

### Migration — Architecture

Replace:
```rust
// old
use ds_api::DeepseekConversation;
let conv = DeepseekConversation::new(client);
```
with:
```rust
// new
use ds_api::Conversation;
let conv = Conversation::new(client);
```

### Migration — Summarizer

Replace:
```rust
// old
use ds_api::TokenBasedSummarizer;

agent.with_summarizer(TokenBasedSummarizer {
    threshold: 60_000,
    retain_last: 10,
    ..Default::default()
})
```
with one of:
```rust
// new — semantic LLM summary (default)
use ds_api::{ApiClient, LlmSummarizer};

agent.with_summarizer(
    LlmSummarizer::new(ApiClient::new(&token))
        .token_threshold(60_000)
        .retain_last(10),
)
```
```rust
// new — sliding window, no extra API calls
use ds_api::SlidingWindowSummarizer;

agent.with_summarizer(SlidingWindowSummarizer::new(20))
```

## [0.3.2] - 2026-03-01

### Summary
This is a patch release that improves the token estimation heuristic, updates documentation and examples, and bumps the crate version to `0.3.2`.

### Changes
- Bumped crate version to `0.3.2`.
- Improved token estimation:
  - Adjusted the chars-to-token heuristic to better handle multibyte characters and edge cases.
  - Fixed an off-by-one rounding issue in the estimator.
- Documentation updates:
  - Updated README and release notes to mention the token estimator improvement and version bump.
  - Ensured examples reference the correct behavior and version.
- Packaging:
  - `ds-api/Cargo.toml` version updated to `0.3.2`.

### Notes
- This release contains no public API changes; it is safe for downstream users (semver patch).
- Recommended checks before publishing:
  - Run `cargo test --manifest-path ds-api/Cargo.toml`
  - Run `cargo clippy -p ds-api -- -D warnings`
  - Run `cargo package --manifest-path ds-api/Cargo.toml`


## [0.3.0] - 2026-02-28

### Summary
This release is a refactor-and-improve release that focuses on:
- Modularization and code hygiene (split large modules into focused submodules).
- English documentation and doc-comments across the crate.
- Observability: tracing instrumentation added to critical API paths.
- Usable examples: runnable example(s) in `examples/` that demonstrate agent + tool flows.
- Linting and tests: Clippy warnings resolved and unit + doctests passing.

### Highlights
- Refactor: Split large modules into smaller submodules for `api/`, `agent/`, `conversation/`, and `raw/`.
- Docs: Translated remaining Chinese inline comments and Rustdoc comments to English across `src/`.
- Examples: Added a runnable `examples/agent_demo.rs` that demonstrates registering a tool and streaming agent events.
- Observability: Added structured tracing calls to `ApiClient` critical paths (request send, streaming, parsing).
- Linting: Clippy issues addressed; the repository compiles cleanly under `-D warnings`.
- Tests: All existing unit tests and doctests pass locally at the time of preparing this release notes.

### Breaking changes
This release includes intentional breaking changes from earlier 0.x versions:
- `Request` and `DeepseekClient` were removed. Use `ApiRequest` and `ApiClient` instead.
- `NormalChatter` and `SimpleChatter` have been removed. Use `DeepseekConversation` and `DeepseekAgent`.
- The `Model` enum is no longer exported as a top-level public type. Use `ApiRequest::deepseek_chat(...)` or `ApiRequest::deepseek_reasoner(...)` to choose a model.
- Public signatures for some types were reorganized through module splitting; consumer code that referenced internal file paths may need to adjust imports to the new layout.

See "Migration notes" below for examples.

### Migration notes
- Replace:
```rust
// old
let req = Request::basic_query(...);
let client = DeepseekClient::new(token);
```
with:
```rust
use ds_api::{ApiClient, ApiRequest};
let client = ApiClient::new(token);
let req = ApiRequest::deepseek_chat(messages).max_tokens(150);
let resp = client.send(req).await?;
```

- Replace:
```rust
// old
let chatter = NormalChatter::new(...);
```
with:
```rust
use ds_api::DeepseekConversation;
let conv = DeepseekConversation::new(client.clone());
```

- Tools:
  - Tools are declared with the `#[tool]` macro and implement the `Tool` trait. Register with `DeepseekAgent::add_tool`.
  - Agent streaming yields two-phase `AgentResponse` events: first preview (assistant content + tool call requests), then the tool results.

### Observability / logging
- `tracing` and `tracing-subscriber` added as optional dependencies to enable structured logging.
- `ApiClient` emits spans and events for:
  - request start (URL + method),
  - applied timeout,
  - HTTP response receive,
  - stream connection and chunk parsing,
  - JSON parsing errors and non-success responses.
- NOTE: Library does NOT automatically install a global tracing subscriber. App binaries/examples should initialize a subscriber (for example: `tracing_subscriber::fmt::init()` or a configured subscriber).

### Examples
- `examples/agent_demo.rs` — runnable demonstration of an agent registering a `WeatherTool` and consuming the agent stream. Run:
```bash
cargo run --example agent_demo --manifest-path ds-api/Cargo.toml
```
Ensure `DEEPSEEK_API_KEY` is set in your environment.

### Internal changes / developer notes
- `src/raw` reorganized into `request/` and `response/` submodules with doctests fixed.
- `agent` split into `agent_core.rs` and `stream.rs` (stream state machine logic isolated).
- `api` reorganized into `client.rs` and `request.rs`.
- Removed obsolete files and renamed `tool.rs` → `tool_trait.rs` (public trait export preserved).
- Many internal imports and visibility specifiers adjusted; public re-exports in `lib.rs` are kept to reduce churn for users.

### Tests
- Unit tests and doctests were updated/moved with the refactor and verify:
  - serialization/ deserialization of `raw` types,
  - conversation/summarizer unit tests,
  - basic agent flow tests.
- All tests pass locally at the time of preparing this release notes.
