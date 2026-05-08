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

pub mod anthropic;

pub use anthropic::AnthropicServer;
pub use anthropic::UpstreamSpec;
pub use anthropic::error::ServerError;
