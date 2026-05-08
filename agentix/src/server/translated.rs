//! Format-agnostic intermediate produced by every inbound translator.
//!
//! Both `server::anthropic::inbound` and `server::openai_chat::inbound`
//! produce a `Translated` value, which `server::fallback` then turns into one
//! agentix `Request` per upstream attempt.

use crate::raw::shared::ToolDefinition;
use crate::request::{Message, ReasoningEffort, ToolChoice};

#[derive(Debug, Clone)]
pub struct Translated {
    pub system_prompt: Option<String>,
    pub model_from_client: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<ToolChoice>,
    pub temperature: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub stream: bool,
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}
