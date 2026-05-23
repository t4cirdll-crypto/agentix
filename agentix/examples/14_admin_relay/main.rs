//! A fully-deployable relay: bearer-token validation + per-user usage log +
//! HTTP Basic-gated admin dashboard. Copy this directory to your own crate
//! as a starting point.
//!
//! Demonstrates how to compose agentix's library primitives:
//!   - The three server modules (Anthropic Messages / OpenAI Chat / Responses)
//!     mounted on one bind, sharing one upstream fallback chain.
//!   - A token registry loaded from `tokens.toml` driving an axum middleware
//!     that gates `/v1/*` and attaches `AuthedUser` to request extensions.
//!   - A `UsageLogger` capturing one JSON-lines record per completed request
//!     (incl. resolved user name).
//!   - An admin router under `/admin` gated by HTTP Basic; reads the usage
//!     log and renders an embedded dashboard (Tailwind + Chart.js via CDN).
//!
//! Run:
//!
//! ```bash
//! cat >tokens.toml <<EOF
//! [[token]]
//! token = "sk-relay-alice"
//! user  = "alice"
//! EOF
//!
//! cargo run --example 14_admin_relay \
//!     --features "server-anthropic,server-openai-chat,server-openai-responses,claude-code" \
//!     -- 127.0.0.1:7878
//! # env: TOKENS_FILE, ADMIN_PASSWORD, USAGE_LOG, AAAGW_UPSTREAM
//! ```

mod admin;
mod aggregate;
mod auth;
mod tokens;

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use agentix::Provider;
use agentix::server::{AnthropicServer, UpstreamSpec, UsageLogger};

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();

    // ── Config (positional addr + env vars) ───────────────────────────
    let listen: String = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:7878".to_string());

    let tokens_path = env_required("TOKENS_FILE");
    let admin_password = env_required("ADMIN_PASSWORD");
    let usage_log_path = env_required("USAGE_LOG");

    if tokens_path.is_none() || admin_password.is_none() || usage_log_path.is_none() {
        eprintln!(
            "set env vars: TOKENS_FILE=tokens.toml ADMIN_PASSWORD=<pwd> USAGE_LOG=/path.jsonl"
        );
        return ExitCode::from(2);
    }
    let tokens_path = tokens_path.unwrap();
    let admin_password = admin_password.unwrap();
    let usage_log_path = usage_log_path.unwrap();

    // ── Upstream chain (single upstream by default; tweak this struct
    //    or wire in CLI parsing for multi-upstream like aaagw does) ───
    let upstream = std::env::var("AAAGW_UPSTREAM").unwrap_or_else(|_| "claude-code".into());
    let chain = match build_chain(&upstream) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("upstream config: {e}");
            return ExitCode::from(2);
        }
    };

    // ── Load tokens.toml ──────────────────────────────────────────────
    let token_registry = match tokens::TokenRegistry::from_file(&tokens_path) {
        Ok(r) => {
            tracing::info!(path = %tokens_path, count = r.len(), "tokens loaded");
            r
        }
        Err(e) => {
            eprintln!("failed to read {tokens_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    // ── Open usage log ────────────────────────────────────────────────
    let usage_logger = match UsageLogger::open(&usage_log_path, true) {
        Ok(l) => Arc::new(l),
        Err(e) => {
            eprintln!("failed to open usage log {usage_log_path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    // ── Build the merged router ───────────────────────────────────────
    let mut router = axum::Router::new();

    let anthropic =
        AnthropicServer::new(chain.clone()).with_usage_logger(usage_logger.clone());
    router = router.merge(anthropic.router());

    #[cfg(feature = "server-openai-chat")]
    {
        use agentix::server::OpenAIChatServer;
        let openai =
            OpenAIChatServer::new(chain.clone()).with_usage_logger(usage_logger.clone());
        router = router.merge(openai.router());
    }

    #[cfg(feature = "server-openai-responses")]
    {
        use agentix::server::OpenAIResponsesServer;
        let resp =
            OpenAIResponsesServer::new(chain.clone()).with_usage_logger(usage_logger.clone());
        router = router.merge(resp.router());
    }

    // Token-auth layer applied to all /v1/* routes registered above.
    let auth_layer = auth::token_auth_layer(token_registry.clone());
    router = router.layer(axum::middleware::from_fn(auth_layer));

    // /admin routes (HTTP Basic).
    let admin_server =
        admin::AdminServer::new(usage_log_path.clone(), admin_password, token_registry);
    router = router.merge(admin_server.router());

    // ── Bind + serve ──────────────────────────────────────────────────
    let listener = match tokio::net::TcpListener::bind(listen.as_str()).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {listen}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let local = listener.local_addr().unwrap();
    tracing::info!(%local, "14_admin_relay listening");

    let serve = axum::serve(listener, router).with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown");
    });
    if let Err(e) = serve.await {
        eprintln!("server: {e}");
        return ExitCode::FAILURE;
    }
    let _ = usage_logger;
    ExitCode::SUCCESS
}

fn env_required(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|s| !s.is_empty())
}

/// Minimal upstream config — provider shortname only. For multi-upstream
/// fallback, copy the argv walker from `src/bin/aaagw.rs`.
fn build_chain(name: &str) -> Result<Vec<UpstreamSpec>, String> {
    let provider = match name {
        "anthropic" => Provider::Anthropic,
        "deepseek" => Provider::DeepSeek,
        "openai" => Provider::OpenAI,
        #[cfg(feature = "claude-code")]
        "claude-code" => Provider::ClaudeCode,
        #[cfg(feature = "codex")]
        "codex" => Provider::Codex,
        other => return Err(format!("unknown upstream: {other}")),
    };

    let token_env = match provider {
        Provider::Anthropic => "ANTHROPIC_API_KEY",
        Provider::DeepSeek => "DEEPSEEK_API_KEY",
        Provider::OpenAI => "OPENAI_API_KEY",
        _ => "",
    };
    let token = if token_env.is_empty() {
        String::new()
    } else {
        std::env::var(token_env).unwrap_or_default()
    };

    let mut spec = UpstreamSpec::new(provider, token);
    spec.pre_commit_timeout = Duration::from_secs(30);
    Ok(vec![spec])
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,agentix=info"));
    let _ = fmt().with_env_filter(filter).try_init();
}
