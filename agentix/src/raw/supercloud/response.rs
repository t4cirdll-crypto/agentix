//! NDJSON response types for the SuperCloud (Render proxy) API.
//!
//! The proxy returns one JSON object per line (newline-delimited JSON / NDJSON).
//! Each line has a `type` field:
//!
//! - `"text"`       → text content delta
//! - `"reasoning"`  → reasoning/thinking content delta
//! - `"tool-call"`  → a complete tool call request
//! - `"finish"`     → stream end with usage and finish reason

use serde::Deserialize;

// ── NDJSON Event ──────────────────────────────────────────────────────────────

/// A single NDJSON line from the proxy response.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(crate) enum NdjsonEvent {
    /// Text content delta.
    Text {
        content: String,
    },
    /// Reasoning/thinking content delta.
    Reasoning {
        content: String,
    },
    /// A complete tool call request.
    ToolCall {
        #[serde(rename = "toolName")]
        tool_name: String,
        args: serde_json::Value,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
    },
    /// Stream termination with usage and reason.
    Finish {
        #[serde(default)]
        reason: String,
        #[serde(default)]
        usage: Option<UsageWire>,
    },
}

/// Usage statistics from the finish event.
/// Fields use camelCase to match the proxy JSON.
#[derive(Debug, Deserialize, Default)]
#[allow(non_snake_case, dead_code)]
pub(crate) struct UsageWire {
    #[serde(default)]
    pub input_tokens: Option<usize>,
    #[serde(rename = "inputTokenDetails")]
    pub input_token_details: Option<InputTokenDetails>,
    #[serde(default)]
    pub output_tokens: Option<usize>,
    #[serde(rename = "outputTokenDetails")]
    pub output_token_details: Option<OutputTokenDetails>,
    #[serde(default)]
    pub total_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
pub(crate) struct InputTokenDetails {
    #[serde(default)]
    pub no_cache_tokens: Option<usize>,
    #[serde(default)]
    pub cache_read_tokens: Option<usize>,
    #[serde(default)]
    pub cache_write_tokens: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
pub(crate) struct OutputTokenDetails {
    #[serde(default)]
    pub text_tokens: Option<usize>,
    #[serde(default)]
    pub reasoning_tokens: Option<usize>,
}
