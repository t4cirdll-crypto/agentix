/*!
agentix — Multi-provider LLM client for Rust.

Supports DeepSeek, OpenAI, Anthropic, Gemini, Kimi, GLM, MiniMax, Mimo,
Grok, and OpenRouter out of the box.
The core API is a value-type [`Request`] that carries everything needed to
hit an LLM API — provider, credentials, model, messages, tools, and tuning.
Call [`Request::stream`] or [`Request::complete`] with a shared `reqwest::Client`.

# Quickstart

```no_run
use agentix::{Request, Provider, Message, UserContent, LlmEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let http = reqwest::Client::new();

    let mut stream = Request::new(Provider::DeepSeek, std::env::var("DEEPSEEK_API_KEY")?)
        .system_prompt("You are helpful.")
        .user("Hello!")
        .stream(&http)
        .await?;

    while let Some(event) = stream.next().await {
        match event {
            LlmEvent::Token(t) => print!("{t}"),
            _                  => {}
        }
    }
    Ok(())
}
```
*/

pub(crate) mod config;
pub mod error;
pub mod msg;
pub(crate) mod provider;
pub mod raw;
pub mod request;
pub mod tool_trait;
pub mod types;

pub mod agent;
#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;
#[cfg(feature = "server-anthropic")]
pub mod server;

// ── Public API ────────────────────────────────────────────────────────────────

pub use agent::{AgentEvent, AgentTurnsStream, agent, agent_turns};
pub use error::ApiError;
pub use msg::LlmEvent;
pub use raw::shared::ToolDefinition;
pub use request::{
    Content, DocumentContent, DocumentData, ImageContent, ImageData, Message, Provider,
    ReasoningEffort, Request, ResponseFormat, ToolCall, ToolChoice, UserContent,
    truncate_to_token_budget,
};
pub use tool_trait::{Tool, ToolBundle, ToolOutput};
pub use types::{CompleteResponse, FinishReason, UsageStats};

pub use agentix_macros::tool;
pub use async_trait;
pub use futures;
pub use schemars;
pub use serde;
pub use serde_json;

#[cfg(feature = "sensitive-logs")]
fn sensitive_logs_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("AGENTIX_LOG_BODIES")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

#[cfg(not(feature = "sensitive-logs"))]
#[allow(dead_code)]
fn sensitive_logs_enabled() -> bool {
    false
}

#[cfg(feature = "mcp")]
pub use mcp::McpTool;
#[cfg(feature = "mcp-server")]
pub use mcp_server::{McpServer, McpServerError, McpService};
