//! OpenAI Chat Completions wire schema. Both `Serialize` and `Deserialize`,
//! since this module is the source of truth for both inbound parsing and
//! outbound construction.
//!
//! Field naming matches OpenAI's API; passes through unknown extra fields to
//! be tolerant of provider extensions.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// OpenAI's newer `max_completion_tokens` (replaces `max_tokens` for o1+
    /// models). We accept it on input, prefer the larger of the two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Catch-all for `top_p`, `top_k`, `stop`, `n`, `seed`,
    /// `presence_penalty`, `frequency_penalty`, `logit_bias`, `user`,
    /// provider-specific extensions, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

// ── Messages ─────────────────────────────────────────────────────────────────

/// Tagged on `role`. Note that `content` shape varies by role:
/// - `system`/`user`: string OR array of content parts
/// - `assistant`: optional string (may be null when only tool_calls present);
///   accompanied by optional `tool_calls` and `reasoning_content` fields
/// - `tool`: string OR array of text parts, paired with `tool_call_id`
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum RequestMessage {
    System {
        content: TextOrParts,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        content: TextOrParts,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<TextOrParts>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCallOnMessage>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Result of a previous tool invocation. `content` is the tool's output;
    /// `tool_call_id` references the assistant's `tool_calls[].id`.
    Tool {
        content: TextOrParts,
        tool_call_id: String,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum TextOrParts {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    /// OpenAI `input_audio` (rare on Chat Completions); preserved through
    /// flatten for tolerance.
    #[serde(other)]
    Other,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCallOnMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCallArgs,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCallArgs {
    pub name: String,
    pub arguments: String,
}

// ── Tools ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDef,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    Named(String), // "auto" | "none" | "required"
    Tool {
        #[serde(rename = "type")]
        kind: String,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: Value },
}

// ── Non-streaming response (server-side construction only) ──────────────────

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseMessage {
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallOnMessage>,
}

// ── Streaming chunk ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<DeltaToolCall>,
}

#[derive(Debug, Serialize, Clone)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "type")]
    pub kind: Option<&'static str>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<DeltaFunctionCall>,
}

#[derive(Debug, Serialize, Clone, Default)]
pub struct DeltaFunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ── Usage ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

impl From<&crate::types::UsageStats> for Usage {
    fn from(u: &crate::types::UsageStats) -> Self {
        Usage {
            prompt_tokens: u.prompt_tokens as u32,
            completion_tokens: u.completion_tokens as u32,
            total_tokens: u.total_tokens as u32,
            prompt_tokens_details: if u.cache_read_tokens > 0 {
                Some(PromptTokensDetails {
                    cached_tokens: u.cache_read_tokens as u32,
                })
            } else {
                None
            },
        }
    }
}
