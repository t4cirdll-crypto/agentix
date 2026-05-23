//! Run an Anthropic Messages-compatible HTTP server backed by an agentix
//! upstream chain. Works with any client that hardcodes Anthropic's API
//! shape (Claude Code, claude-code-router, etc.).
//!
//! Run with:
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example 12_server_anthropic --features server-anthropic
//! ```

use agentix::Provider;
use agentix::server::{AnthropicServer, UpstreamSpec};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "set ANTHROPIC_API_KEY to run this example")?;

    let chain = vec![UpstreamSpec::new(Provider::Anthropic, key)];
    AnthropicServer::new(chain).listen("127.0.0.1:7878").await?;
    Ok(())
}
