//! Unified request layer.
//!
//! [`Request`] is the self-contained, provider-agnostic chat-completion request.
//! It carries *everything* needed to hit an LLM API — provider, credentials,
//! model, messages, tools, and tuning knobs.
//!
//! ```no_run
//! use agentix::{Request, Provider, Message, UserContent, LlmEvent};
//! use futures::StreamExt;
//!
//! # #[tokio::main] async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let http = reqwest::Client::new();
//!
//! let mut stream = Request::new(Provider::DeepSeek, "sk-...")
//!     .model("deepseek-chat")
//!     .system_prompt("You are helpful.")
//!     .user("Hello!")
//!     .stream(&http)
//!     .await?;
//!
//! while let Some(event) = stream.next().await {
//!     if let LlmEvent::Token(t) = event { print!("{t}"); }
//! }
//! # Ok(()) }
//! ```

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::raw::shared::ToolDefinition;
use crate::types::CompleteResponse;

// ─── Message ────────────────────────────────────────────────────────────────

/// Image content that can be embedded in a user message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageContent {
    /// The image payload.
    pub data: ImageData,
    /// MIME type, e.g. `"image/jpeg"`, `"image/png"`.
    pub mime_type: String,
}

/// How the image data is provided.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ImageData {
    /// Base64-encoded image bytes.
    Base64(String),
    /// Publicly accessible URL.
    Url(String),
}

/// Document (e.g. PDF) content that can be embedded in a user message.
///
/// Supported providers: Anthropic (`document` block), OpenAI Responses API
/// (`input_file`), Gemini (`inline_data` / `file_data`), OpenRouter (`file`
/// plugin). Non-multimodal OpenAI-compatible providers (DeepSeek, Grok,
/// Kimi, GLM) silently drop document parts on request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentContent {
    /// The document payload.
    pub data: DocumentData,
    /// MIME type, e.g. `"application/pdf"`.
    pub mime_type: String,
    /// Optional filename. OpenAI's `input_file` requires a filename when
    /// using `file_data`; if absent the provider adapter supplies a generic
    /// placeholder (`"document.pdf"` for PDFs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// How the document data is provided.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentData {
    /// Base64-encoded document bytes.
    Base64(String),
    /// Publicly accessible URL.
    Url(String),
}

/// A single content block — used in user messages, tool results, and tool outputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text { text: String },
    Image(ImageContent),
    Document(DocumentContent),
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Content::Text { text: s.into() }
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Text {
            text: s.to_string(),
        }
    }
}
impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Text { text: s }
    }
}

/// Backwards-compatible type alias.
pub type UserContent = Content;

/// A single turn in a conversation. Every variant carries exactly the fields
/// it needs — no invalid states are representable.
#[derive(Debug, Clone)]
pub enum Message {
    /// A message from the human side, supporting text and images.
    User(Vec<UserContent>),

    /// A message produced by the model. `content` and `tool_calls` may both be
    /// present; `content` may be absent when the model only emits tool calls.
    Assistant {
        content: Option<String>,
        /// Provider-specific chain-of-thought / reasoning text, if any.
        reasoning: Option<String>,
        tool_calls: Vec<ToolCall>,
        /// Opaque per-turn state emitted by the provider (e.g. Anthropic
        /// thinking blocks with signatures). Populated from
        /// [`LlmEvent::AssistantState`] and round-tripped verbatim by the
        /// same provider's request serializer. Always `None` for providers
        /// that don't emit state.
        provider_data: Option<serde_json::Value>,
    },

    /// The result of a tool invocation, keyed by the call's ID.
    ToolResult {
        call_id: String,
        content: Vec<Content>,
    },
}

impl Message {
    /// Estimate the number of tokens in this message using tiktoken.
    ///
    /// Note: This is an estimation. Different providers have slightly different
    /// tokenization rules and overheads for message metadata (role, name, etc.).
    pub fn estimate_tokens(&self) -> usize {
        use std::sync::OnceLock;
        static BPE: OnceLock<tiktoken_rs::CoreBPE> = OnceLock::new();
        let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().unwrap());
        let mut tokens = 0;

        match self {
            Message::User(parts) => {
                tokens += 4; // overhead for role
                for part in parts {
                    match part {
                        UserContent::Text { text: t } => {
                            tokens += bpe.encode_with_special_tokens(t).len()
                        }
                        UserContent::Image(_) => tokens += 1000, // rough fixed cost for images
                        UserContent::Document(_) => tokens += 2000, // rough fixed cost for docs
                    }
                }
            }
            Message::Assistant {
                content,
                reasoning,
                tool_calls,
                ..
            } => {
                tokens += 4;
                if let Some(c) = content {
                    tokens += bpe.encode_with_special_tokens(c).len();
                }
                if let Some(r) = reasoning {
                    tokens += bpe.encode_with_special_tokens(r).len();
                }
                for tc in tool_calls {
                    tokens += bpe.encode_with_special_tokens(&tc.name).len();
                    tokens += bpe.encode_with_special_tokens(&tc.arguments).len();
                }
            }
            Message::ToolResult { content, .. } => {
                tokens += 4;
                for part in content {
                    match part {
                        Content::Text { text } => {
                            tokens += bpe.encode_with_special_tokens(text).len()
                        }
                        Content::Image(_) => tokens += 1000,
                        Content::Document(_) => tokens += 2000,
                    }
                }
            }
        }
        tokens
    }
}

/// Drop the oldest messages from `history` until the total estimated token
/// count is at or below `budget`.  Always keeps at least one message.
///
/// After finding the cut point, advances it forward past any leading
/// `ToolResult` messages so that `Assistant { tool_calls }` / `ToolResult`
/// pairs are never split — which would cause provider errors.
pub fn truncate_to_token_budget(history: &mut Vec<Message>, budget: usize) {
    // Scan from the back, accumulating tokens until we exceed the budget.
    let mut acc: usize = 0;
    let mut keep_from = history.len(); // default: keep all (no truncation needed)
    for (i, msg) in history.iter().enumerate().rev() {
        acc += msg.estimate_tokens();
        if acc > budget {
            keep_from = (i + 1).min(history.len() - 1);
            break;
        }
    }

    // If keep_from == history.len(), history fits within budget — nothing to do.
    if keep_from == history.len() {
        return;
    }

    // Advance keep_from past any leading ToolResult messages so we never
    // start the history mid-tool-call-group (orphaned tool results).
    while keep_from < history.len() {
        match &history[keep_from] {
            Message::ToolResult { .. } => keep_from += 1,
            _ => break,
        }
    }

    // Also skip past an Assistant-with-tool-calls whose ToolResults follow,
    // otherwise those ToolResults would become orphaned after the drain.
    if keep_from < history.len()
        && let Message::Assistant { tool_calls, .. } = &history[keep_from]
        && !tool_calls.is_empty()
    {
        // Collect the tool_call ids that belong to this assistant turn.
        let ids: std::collections::HashSet<&str> =
            tool_calls.iter().map(|tc| tc.id.as_str()).collect();
        keep_from += 1;
        // Skip all consecutive ToolResult messages that belong to this group.
        while keep_from < history.len() {
            match &history[keep_from] {
                Message::ToolResult { call_id, .. } if ids.contains(call_id.as_str()) => {
                    keep_from += 1;
                }
                _ => break,
            }
        }
    }

    // Safety: always keep at least one message.
    if keep_from >= history.len() {
        keep_from = history.len().saturating_sub(1);
    }

    if keep_from > 0 {
        history.drain(0..keep_from);
    }
}

/// A single tool invocation requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call ID (used to match results back).
    pub id: String,
    /// Name of the tool the model wants to invoke.
    pub name: String,
    /// Raw JSON string produced by the model.
    pub arguments: String,
}

// ─── Provider ──────────────────────────────────────────────────────────────

/// Which LLM provider to use.
///
/// Each variant determines the request/response format, auth method, and
/// default base URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    #[serde(rename = "deepseek")]
    DeepSeek,
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
    /// Moonshot AI (Kimi)
    #[serde(rename = "kimi")]
    Kimi,
    /// Zhipu AI (ChatGLM)
    #[serde(rename = "glm")]
    Glm,
    /// MiniMax
    #[serde(rename = "minimax")]
    Minimax,
    /// Xiaomi MiMo (Anthropic-compatible Messages API)
    #[serde(rename = "mimo")]
    Mimo,
    /// xAI Grok
    #[serde(rename = "grok")]
    Grok,
    /// OpenRouter (API gateway with prompt caching support)
    #[serde(rename = "openrouter")]
    OpenRouter,
    /// Claude Code CLI — rides Max OAuth via the local `claude` binary.
    /// The `api_key` field is ignored; auth comes from the CLI's keychain.
    #[cfg(feature = "claude-code")]
    #[serde(rename = "claude-code")]
    ClaudeCode,
    /// Codex CLI via `codex app-server`.
    /// The `api_key` field is ignored; auth comes from the local Codex CLI.
    #[cfg(feature = "codex")]
    #[serde(rename = "codex")]
    Codex,
    /// SuperCloud (Render.com proxy for ConcentrateAI) — uses NDJSON format
    /// with OAuth bearer token from `~/.better-auth/token.json`.
    #[serde(rename = "supercloud")]
    SuperCloud,
}

impl Provider {
    /// Default base URL for this provider.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Provider::DeepSeek => "https://api.deepseek.com",
            Provider::OpenAI => "https://api.openai.com/v1",
            Provider::Anthropic => "https://api.anthropic.com",
            Provider::Gemini => "https://generativelanguage.googleapis.com/v1beta",
            Provider::Kimi => "https://api.moonshot.cn/v1",
            Provider::Glm => "https://open.bigmodel.cn/api/paas/v4",
            Provider::Minimax => "https://api.minimaxi.com/anthropic",
            Provider::Mimo => "https://api.xiaomimimo.com/anthropic",
            Provider::Grok => "https://api.x.ai/v1",
            Provider::OpenRouter => "https://openrouter.ai/api/v1",
            #[cfg(feature = "claude-code")]
            Provider::ClaudeCode => "",
            #[cfg(feature = "codex")]
            Provider::Codex => "",
            Provider::SuperCloud => "https://supercode-8w7e.onrender.com",
        }
    }

    /// Default model for this provider.
    pub fn default_model(&self) -> &'static str {
        match self {
            Provider::DeepSeek => "deepseek-chat",
            Provider::OpenAI => "gpt-4o",
            Provider::Anthropic => "claude-sonnet-4-20250514",
            Provider::Gemini => "gemini-2.0-flash",
            Provider::Kimi => "kimi-k2.5",
            Provider::Glm => "glm-5",
            Provider::Minimax => "MiniMax-M2.7",
            Provider::Mimo => "mimo-v2.5-pro",
            Provider::Grok => "grok-4",
            Provider::OpenRouter => "openrouter/auto",
            #[cfg(feature = "claude-code")]
            Provider::ClaudeCode => "sonnet",
            #[cfg(feature = "codex")]
            Provider::Codex => "gpt-5.5",
            Provider::SuperCloud => "deepseek-v4-flash",
        }
    }
}

// ─── ReasoningEffort ────────────────────────────────────────────────────────

/// Cross-provider hint for how much compute to spend on internal reasoning.
///
/// `None` explicitly disables thinking on providers that support a disable
/// toggle (DeepSeek, Anthropic adaptive). Individual providers coerce the
/// remaining levels to their own scale — e.g. DeepSeek only has `high`/`max`,
/// so `Minimal/Low/Medium` collapse to `High` and `XHigh` collapses to `Max`;
/// Anthropic has no `minimal`, so `Minimal` collapses to `low`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

// ─── ToolChoice ─────────────────────────────────────────────────────────────

/// Provider-agnostic tool selection hint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide (default when tools are present).
    #[default]
    Auto,
    /// The model must not call any tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// Force a specific tool by name.
    Tool(String),
}

// ─── Request ────────────────────────────────────────────────────────────────

/// A self-contained, provider-agnostic chat-completion request.
///
/// Carries everything needed to hit an LLM API: provider, credentials, model,
/// messages, tools, and tuning parameters.
///
/// Call [`stream()`][Request::stream] or [`complete()`][Request::complete] with
/// a shared `reqwest::Client` to send the request.
#[derive(Debug, Clone)]
pub struct Request {
    // ── Identity ──────────────────────────────────────────────────────────
    /// Which provider to use.
    pub provider: Provider,
    /// API key / token.
    pub api_key: String,
    /// Base URL override. If empty, uses [`Provider::default_base_url`].
    pub base_url: String,

    // ── Model & messages ─────────────────────────────────────────────────
    /// Model identifier (e.g. `"deepseek-chat"`, `"gpt-4o"`).
    pub model: String,
    /// Optional system prompt.
    pub system_message: Option<String>,
    /// Conversation history.
    pub messages: Vec<Message>,

    // ── Tools ────────────────────────────────────────────────────────────
    /// Tools the model may call.
    pub tools: Vec<ToolDefinition>,
    /// How the model should select tools.
    pub tool_choice: Option<ToolChoice>,

    // ── Tuning ───────────────────────────────────────────────────────────
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Reasoning effort hint. `None` means leave provider default in place;
    /// `Some(ReasoningEffort::None)` explicitly disables thinking on providers
    /// that support it.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Constrain the output format.
    pub response_format: Option<ResponseFormat>,
    /// Arbitrary extra top-level JSON fields merged into the provider's
    /// raw request body (e.g. `prefix`, `thinking`).
    pub extra_body: serde_json::Map<String, serde_json::Value>,

    // ── Retry ────────────────────────────────────────────────────────────
    /// Maximum retries for transient errors. Default: 3.
    pub max_retries: u32,
    /// Initial retry delay in milliseconds. Default: 1000.
    pub retry_delay_ms: u64,
}

impl Request {
    /// Create a new request for the given provider and API key.
    ///
    /// Sets sensible defaults: provider's default base URL and model,
    /// 3 retries with 1 s initial delay, no system prompt.
    pub fn new(provider: Provider, api_key: impl Into<String>) -> Self {
        Self {
            base_url: provider.default_base_url().to_string(),
            model: provider.default_model().to_string(),
            api_key: api_key.into(),
            provider,
            system_message: None,
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            temperature: None,
            reasoning_effort: None,
            max_tokens: None,
            response_format: None,
            extra_body: serde_json::Map::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }

    /// Shortcut for `Request::new(Provider::DeepSeek, api_key)`.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::new(Provider::DeepSeek, api_key)
    }

    /// Shortcut for `Request::new(Provider::OpenAI, api_key)`.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(Provider::OpenAI, api_key)
    }

    /// Shortcut for `Request::new(Provider::Anthropic, api_key)`.
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::new(Provider::Anthropic, api_key)
    }

    /// Shortcut for `Request::new(Provider::Gemini, api_key)`.
    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::new(Provider::Gemini, api_key)
    }

    /// Shortcut for `Request::new(Provider::Kimi, api_key)`.
    pub fn kimi(api_key: impl Into<String>) -> Self {
        Self::new(Provider::Kimi, api_key)
    }

    /// Shortcut for `Request::new(Provider::Glm, api_key)`.
    pub fn glm(api_key: impl Into<String>) -> Self {
        Self::new(Provider::Glm, api_key)
    }

    /// Shortcut for `Request::new(Provider::Minimax, api_key)`.
    pub fn minimax(api_key: impl Into<String>) -> Self {
        Self::new(Provider::Minimax, api_key)
    }

    /// Shortcut for `Request::new(Provider::Mimo, api_key)`.
    pub fn mimo(api_key: impl Into<String>) -> Self {
        Self::new(Provider::Mimo, api_key)
    }

    /// Shortcut for `Request::new(Provider::Grok, api_key)`.
    pub fn grok(api_key: impl Into<String>) -> Self {
        Self::new(Provider::Grok, api_key)
    }

    /// Shortcut for `Request::new(Provider::OpenRouter, api_key)`.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::new(Provider::OpenRouter, api_key)
    }

    /// Shortcut for `Request::new(Provider::ClaudeCode, "")`.
    ///
    /// No API key is required — auth is delegated to the local `claude` CLI.
    #[cfg(feature = "claude-code")]
    pub fn claude_code() -> Self {
        Self::new(Provider::ClaudeCode, String::new())
    }

    /// Shortcut for `Request::new(Provider::Codex, "")`.
    ///
    /// No API key is required; auth is delegated to the local `codex` CLI.
    #[cfg(feature = "codex")]
    pub fn codex() -> Self {
        Self::new(Provider::Codex, String::new())
    }

    /// Shortcut for `Request::new(Provider::SuperCloud, api_key)`.
    ///
    /// The `api_key` is the OAuth bearer token from `~/.better-auth/token.json`.
    pub fn supercloud(api_key: impl Into<String>) -> Self {
        Self::new(Provider::SuperCloud, api_key)
    }

    // ── Builder setters (all consume & return Self) ──────────────────────

    /// Override the base URL.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model.
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = m.into();
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_message = Some(p.into());
        self
    }

    /// Append a message to the conversation.
    pub fn message(mut self, m: Message) -> Self {
        self.messages.push(m);
        self
    }

    /// Append a user text message (convenience).
    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(Message::User(vec![Content::text(text)]))
    }

    /// Set the full message history.
    pub fn messages(mut self, msgs: Vec<Message>) -> Self {
        self.messages = msgs;
        self
    }

    /// Set the tool definitions.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set how the model should select tools.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Set the max tokens.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Set the reasoning-effort hint. `ReasoningEffort::None` explicitly
    /// disables thinking on providers that support the toggle.
    pub fn reasoning_effort(mut self, e: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(e);
        self
    }

    /// Set the response format to plain text (the default).
    pub fn text(mut self) -> Self {
        self.response_format = Some(ResponseFormat::Text);
        self
    }

    /// Constrain output to a named JSON Schema (OpenAI `json_schema` mode).
    ///
    /// Use `schemars::schema_for!(T)` to generate the schema:
    /// ```ignore
    /// let schema = serde_json::to_value(schemars::schema_for!(MyStruct)).unwrap();
    /// let req = Request::openai(key).json_schema("my_struct", schema, true);
    /// ```
    pub fn json_schema(
        mut self,
        name: impl Into<String>,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        self.response_format = Some(ResponseFormat::JsonSchema {
            name: name.into(),
            schema,
            strict,
        });
        self
    }

    /// Set the response format to JSON object mode.
    ///
    /// The model will be constrained to emit a valid JSON object. You must
    /// also instruct the model to produce JSON in your system prompt or user
    /// message — the format flag alone is not sufficient for most providers.
    pub fn json(mut self) -> Self {
        self.response_format = Some(ResponseFormat::JsonObject);
        self
    }

    /// Set retry parameters.
    pub fn retries(mut self, max: u32, initial_delay_ms: u64) -> Self {
        self.max_retries = max;
        self.retry_delay_ms = initial_delay_ms;
        self
    }

    /// Merge extra JSON fields into the request body.
    pub fn extra_body(mut self, extra: serde_json::Map<String, serde_json::Value>) -> Self {
        self.extra_body = extra;
        self
    }

    // ── Effective base URL ───────────────────────────────────────────────

    /// Resolve the effective base URL (custom or provider default).
    pub fn effective_base_url(&self) -> &str {
        if self.base_url.is_empty() {
            self.provider.default_base_url()
        } else {
            &self.base_url
        }
    }

    // ── Send ─────────────────────────────────────────────────────────────

    /// Send a streaming request and return a stream of [`LlmEvent`]s.
    pub async fn stream(
        &self,
        http: &reqwest::Client,
    ) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
        let config = self.to_agent_config();
        let messages = &self.messages;
        let tools = &self.tools;

        match self.provider {
            Provider::DeepSeek => {
                crate::raw::deepseek::stream_deepseek(&self.api_key, http, &config, messages, tools)
                    .await
            }
            Provider::OpenAI => {
                crate::raw::openai::stream_openai(&self.api_key, http, &config, messages, tools)
                    .await
            }
            Provider::Anthropic => {
                crate::raw::anthropic::stream_anthropic(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            Provider::Gemini => {
                crate::raw::gemini::stream_gemini(&self.api_key, http, &config, messages, tools)
                    .await
            }
            Provider::Minimax => {
                crate::raw::anthropic::stream_anthropic(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            Provider::Mimo => {
                crate::raw::mimo::stream_mimo(&self.api_key, http, &config, messages, tools).await
            }
            Provider::Kimi => {
                crate::raw::kimi::stream_kimi(&self.api_key, http, &config, messages, tools).await
            }
            Provider::Glm => {
                crate::raw::glm::stream_glm(&self.api_key, http, &config, messages, tools).await
            }
            Provider::Grok => {
                crate::raw::grok::stream_grok(&self.api_key, http, &config, messages, tools).await
            }
            Provider::OpenRouter => {
                crate::raw::openrouter::stream_openrouter(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            #[cfg(feature = "claude-code")]
            Provider::ClaudeCode => {
                crate::raw::claude_code::stream_claude_code(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            #[cfg(feature = "codex")]
            Provider::Codex => {
                crate::raw::codex::stream_codex(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                    self.tool_choice.as_ref(),
                )
                .await
            }
            Provider::SuperCloud => {
                crate::raw::supercloud::stream_supercloud(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
        }
    }

    /// Send a non-streaming request and return the complete response.
    pub async fn complete(&self, http: &reqwest::Client) -> Result<CompleteResponse, ApiError> {
        let config = self.to_agent_config();
        let messages = &self.messages;
        let tools = &self.tools;

        match self.provider {
            Provider::DeepSeek => {
                crate::raw::deepseek::complete_deepseek(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            Provider::OpenAI => {
                crate::raw::openai::complete_openai(&self.api_key, http, &config, messages, tools)
                    .await
            }
            Provider::Anthropic => {
                crate::raw::anthropic::complete_anthropic(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            Provider::Gemini => {
                crate::raw::gemini::complete_gemini(&self.api_key, http, &config, messages, tools)
                    .await
            }
            Provider::Minimax => {
                crate::raw::anthropic::complete_anthropic(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            Provider::Mimo => {
                crate::raw::mimo::complete_mimo(&self.api_key, http, &config, messages, tools).await
            }
            Provider::Kimi => {
                crate::raw::kimi::complete_kimi(&self.api_key, http, &config, messages, tools).await
            }
            Provider::Glm => {
                crate::raw::glm::complete_glm(&self.api_key, http, &config, messages, tools).await
            }
            Provider::Grok => {
                crate::raw::grok::complete_grok(&self.api_key, http, &config, messages, tools).await
            }
            Provider::OpenRouter => {
                crate::raw::openrouter::complete_openrouter(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            #[cfg(feature = "claude-code")]
            Provider::ClaudeCode => {
                crate::raw::claude_code::complete_claude_code(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
            #[cfg(feature = "codex")]
            Provider::Codex => {
                crate::raw::codex::complete_codex(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                    self.tool_choice.as_ref(),
                )
                .await
            }
            Provider::SuperCloud => {
                crate::raw::supercloud::complete_supercloud(
                    &self.api_key,
                    http,
                    &config,
                    messages,
                    tools,
                )
                .await
            }
        }
    }

    /// Convert to the legacy `AgentConfig` for internal provider use.
    ///
    /// This is a temporary bridge until providers are fully migrated.
    fn to_agent_config(&self) -> crate::config::AgentConfig {
        crate::config::AgentConfig {
            base_url: self.effective_base_url().to_string(),
            model: self.model.clone(),
            system_prompt: self.system_message.clone(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            reasoning_effort: self.reasoning_effort,
            extra_body: self.extra_body.clone(),
            response_format: self.response_format.clone(),
            max_retries: self.max_retries,
            retry_delay_ms: self.retry_delay_ms,
        }
    }
}

#[cfg(test)]
mod truncate_tests {
    use super::*;

    fn user(s: &str) -> Message {
        Message::User(vec![crate::UserContent::Text {
            text: s.repeat(200),
        }])
    }
    fn assistant_text(s: &str) -> Message {
        Message::Assistant {
            content: Some(s.repeat(200)),
            reasoning: None,
            tool_calls: vec![],
            provider_data: None,
        }
    }
    fn assistant_tc(ids: &[&str]) -> Message {
        Message::Assistant {
            content: None,
            reasoning: None,
            tool_calls: ids
                .iter()
                .map(|id| ToolCall {
                    id: id.to_string(),
                    name: "bash".to_string(),
                    arguments: "{}".to_string(),
                })
                .collect(),
            provider_data: None,
        }
    }
    fn tool_result(id: &str) -> Message {
        Message::ToolResult {
            call_id: id.to_string(),
            content: vec![Content::text("ok")],
        }
    }

    fn no_orphans(history: &[Message]) {
        use std::collections::HashSet;
        let called: HashSet<&str> = history
            .iter()
            .filter_map(|m| {
                if let Message::Assistant { tool_calls, .. } = m {
                    Some(tool_calls.iter().map(|tc| tc.id.as_str()))
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        for m in history {
            if let Message::ToolResult { call_id, .. } = m {
                assert!(
                    called.contains(call_id.as_str()),
                    "orphaned ToolResult with call_id={call_id}"
                );
            }
        }
    }

    #[test]
    fn test_no_truncation_needed() {
        let mut h = vec![user("a"), assistant_text("b")];
        truncate_to_token_budget(&mut h, 1_000_000);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_orphaned_tool_results_skipped_at_start() {
        // If keep_from lands on a ToolResult, skip it.
        let mut h = vec![
            user("x"),
            assistant_tc(&["id1"]),
            tool_result("id1"),
            user("y"),
            assistant_text("z"),
        ];
        let budget = h[3..].iter().map(|m| m.estimate_tokens()).sum::<usize>() + 10;
        truncate_to_token_budget(&mut h, budget);
        no_orphans(&h);
        // Should start at user("y") or later — first message must not be a
        // ToolResult (which would be orphaned).
        assert!(!matches!(h.first(), Some(Message::ToolResult { .. })));
        no_orphans(&h);
    }

    #[test]
    fn test_assistant_with_tool_calls_not_split_from_results() {
        // keep_from lands on assistant_tc — its ToolResults must be dropped too.
        let mut h = vec![
            user("old"),
            assistant_tc(&["a1", "a2"]),
            tool_result("a1"),
            tool_result("a2"),
            user("new"),
            assistant_text("reply"),
        ];
        // Budget only fits the last two messages.
        let budget = h[4..].iter().map(|m| m.estimate_tokens()).sum::<usize>() + 10;
        truncate_to_token_budget(&mut h, budget);
        no_orphans(&h);
        // The assistant_tc group and its results must all be gone.
        assert!(!h.iter().any(|m| matches!(m, Message::ToolResult { .. })));
    }

    #[test]
    fn test_always_keeps_at_least_one_message() {
        let mut h = vec![user("only")];
        truncate_to_token_budget(&mut h, 1);
        assert_eq!(h.len(), 1);
    }
}

/// Provider-agnostic output-format hint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    /// Strict JSON Schema output (OpenAI `json_schema` mode).
    /// `schema` should be a `schemars::schema_for!(T)` value serialized to `Value`.
    JsonSchema {
        /// Name for the schema (shown in API responses).
        name: String,
        /// The JSON Schema object.
        schema: serde_json::Value,
        /// Whether to enforce strict schema adherence.
        strict: bool,
    },
}
