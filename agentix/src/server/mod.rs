//! HTTP servers that expose agentix as an LLM-API-compatible endpoint.
//!
//! Following the pandoc model: agentix's internal `Request` + `Message` +
//! `LlmEvent` AST is the hub; the existing 10 outbound providers are
//! "writers"; the modules here are "readers" that accept inbound requests in
//! various wire formats and translate them to the AST.
//!
//! Currently implemented:
//!
//! - [`anthropic`] — Anthropic Messages format on `POST /v1/messages`.
//! - [`openai_chat`] — OpenAI Chat Completions on `POST /v1/chat/completions`
//!   (gated by the `server-openai-chat` feature).
//!
//! All readers share one fallback chain (see [`UpstreamSpec`]) and one shared
//! state, so a single bind port can serve multiple wire formats simultaneously
//! by merging their routers — see the `agentix` CLI binary for an example.

pub mod fallback;
pub mod translated;

pub mod anthropic;

#[cfg(feature = "server-openai-chat")]
pub mod openai_chat;

pub use anthropic::AnthropicServer;
pub use anthropic::error::ServerError;
pub use fallback::UpstreamSpec;
pub use translated::Translated;

#[cfg(feature = "server-openai-chat")]
pub use openai_chat::OpenAIChatServer;
