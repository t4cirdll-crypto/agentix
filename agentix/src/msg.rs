use crate::request::ToolCall;
use crate::types::UsageStats;

// ── LLM Provider Events ──────────────────────────────────────────────────────

/// Raw events emitted by an LLM Provider.
///
/// Marked `#[non_exhaustive]` so new variants can be added without breaking
/// downstream matchers — always include a `_ => { /* ignore */ }` arm.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LlmEvent {
    /// A text fragment.
    Token(String),
    /// A reasoning/thinking fragment.
    Reasoning(String),
    /// Signature attached to the most recently opened reasoning block. Currently
    /// only Anthropic-compatible providers emit this; consumers that don't care
    /// about signature passthrough can ignore the variant.
    ReasoningSignature(String),
    /// A tool call fragment emitted during streaming.
    ToolCallChunk(crate::types::ToolCallChunk),
    /// A tool call requested by the model.
    ToolCall(ToolCall),
    /// Opaque provider-specific per-turn state (e.g. Anthropic thinking
    /// blocks with signatures). Emit once before `Done`; downstream consumers
    /// attach it to the reconstructed `Message::Assistant.provider_data`.
    /// Other providers may never emit this.
    AssistantState(serde_json::Value),
    /// Usage statistics (usually sent at the end).
    Usage(UsageStats),
    /// The stream has ended.
    Done,
    /// A provider-level error.
    Error(String),
}
