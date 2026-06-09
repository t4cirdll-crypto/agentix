use serde_json::{Map, Value};

/// Internal configuration struct bridging [`Request`](crate::Request) to raw
/// provider functions.  Not part of the public API.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Provider API base URL (e.g. `"https://api.deepseek.com"`).
    pub base_url: String,
    /// Model identifier (e.g. `"deepseek-chat"`, `"gpt-4o"`).
    pub model: String,
    /// Optional system prompt prepended to every request.
    pub system_prompt: Option<String>,
    /// Maximum tokens to generate. `None` = provider default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature. `None` = provider default.
    pub temperature: Option<f32>,
    /// Reasoning-effort hint (providers coerce as needed). `None` = default;
    /// `Some(ReasoningEffort::None)` explicitly disables thinking where supported.
    pub reasoning_effort: Option<crate::request::ReasoningEffort>,
    /// Extra JSON fields merged into the request body (provider-specific).
    pub extra_body: Map<String, Value>,
    /// Response format constraint (provider support varies).
    pub response_format: Option<crate::request::ResponseFormat>,

    /// Maximum number of retries for transient errors. Default: 3.
    pub max_retries: u32,
    /// Initial delay between retries in milliseconds. Default: 1000ms.
    pub retry_delay_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            model: String::new(),
            system_prompt: None,
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
            extra_body: Map::new(),
            response_format: None,
            max_retries: 3,
            retry_delay_ms: 1000,
        }
    }
}
