//! Raw, provider-specific request/response types.
//!
//! Each sub-module maps directly to a provider's JSON schema and is used
//! internally by the [`Request`](crate::Request) dispatch layer.
//! The [`shared`] module contains types common across providers (e.g.
//! [`ToolDefinition`](shared::ToolDefinition)).
//!
//! Most users should interact through [`Request`](crate::Request) and never
//! need to touch these types directly.
pub mod anthropic;
#[cfg(feature = "claude-code")]
pub mod claude_code;
#[cfg(feature = "codex")]
pub mod codex;
pub mod deepseek;
pub mod gemini;
pub mod glm;
pub mod grok;
pub mod kimi;
pub mod mimo;
pub mod openai;
pub mod openrouter;
pub mod shared;

pub use shared::{
    FunctionDefinition, FunctionName, JsonSchemaBody, ResponseFormat, ToolChoice,
    ToolChoiceFunction, ToolChoiceMode, ToolDefinition, ToolKind,
};
