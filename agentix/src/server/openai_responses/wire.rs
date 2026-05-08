//! OpenAI Responses API wire schema. Inbound (request) types deserialize
//! cleanly from clients; outbound (response, stream events) types serialize
//! cleanly back. We don't aim for round-trippable Serialize+Deserialize on
//! every type — typed deserialization of inbound items has different
//! constraints than typed serialization of outbound items (encrypted content,
//! IDs the client expects to see verbatim).
//!
//! Key wire facts:
//!   - `input` is either a string OR an array of typed items.
//!   - `instructions` is the system prompt (NOT a "system" message item).
//!   - Tools are flat-typed: `{"type":"function","name":...}` — no `function:` wrapper.
//!   - Function calls and outputs use `call_id` (not `tool_call_id`).
//!   - Reasoning items carry `id`, `summary[]`, optional `encrypted_content`.
//!   - The non-streaming response is a `{"object":"response", "output":[...]}` envelope.
//!   - Streaming has 15+ event types; each has a `type` field.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Clone)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: InputField,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<RequestTool>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub text: Option<TextConfig>,
    /// Default `true`. When true and the request returns successfully, agentix
    /// stores the resolved input + output items so subsequent requests can
    /// reference them via `previous_response_id`.
    #[serde(default = "default_true")]
    pub store: bool,
    /// Chain to a previous response. We resolve to its stored input + output
    /// items and prepend them to this request's input.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub truncation: Option<String>,
    /// Catch-all for top_p, top_logprobs, user, metadata, service_tier, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum InputField {
    Text(String),
    Items(Vec<Value>),
}

// ── Inbound input items (typed view, kept as Value for unknown variants) ────

/// Typed view of an `input[]` item. We accept the named variants and treat
/// everything else as opaque (preserved for round-trip into upstream).
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedInputItem {
    Message {
        role: String,
        content: MessageContent,
        #[serde(default)]
        id: Option<String>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default)]
        id: Option<String>,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
        #[serde(default)]
        id: Option<String>,
    },
    /// Reasoning items — preserve verbatim into provider_data for round-trip.
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<Value>,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<MessageContentPart>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContentPart {
    InputText {
        text: String,
    },
    InputImage {
        #[serde(default)]
        image_url: Option<String>,
        #[serde(default)]
        detail: Option<String>,
    },
    InputFile {
        #[serde(default)]
        file_data: Option<String>,
        #[serde(default)]
        file_url: Option<String>,
        #[serde(default)]
        filename: Option<String>,
    },
    /// Output text appears in assistant message items in input history (for
    /// previous-turn replay). We accept it on inbound for completeness.
    OutputText {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RequestTool {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    Named(String),
    Object(ToolChoiceObject),
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolChoiceObject {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TextConfig {
    #[serde(default)]
    pub format: Option<TextFormat>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    Text,
    JsonObject,
    JsonSchema {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        schema: Option<Value>,
        #[serde(default)]
        strict: Option<bool>,
    },
}

// ── Outbound response object (server-constructed, Serialize-only) ───────────

#[derive(Debug, Serialize, Clone)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: &'static str,
    pub model: String,
    pub output: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub tool_choice: Value,
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    pub parallel_tool_calls: bool,
    pub truncation: &'static str,
    pub usage: Option<Usage>,
    pub metadata: serde_json::Map<String, Value>,
    pub incomplete_details: Option<Value>,
}

#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens_details: OutputTokensDetails,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

impl From<&crate::types::UsageStats> for Usage {
    fn from(u: &crate::types::UsageStats) -> Self {
        Usage {
            input_tokens: u.prompt_tokens as u32,
            output_tokens: u.completion_tokens as u32,
            total_tokens: u.total_tokens as u32,
            input_tokens_details: InputTokensDetails {
                cached_tokens: u.cache_read_tokens as u32,
            },
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: u.reasoning_tokens as u32,
            },
        }
    }
}
