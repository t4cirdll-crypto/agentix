//! A fully-deployable relay: bearer-token validation + per-user usage log +
//! HTTP Basic-gated admin dashboard + multi-upstream per-model routing.
//! Copy this directory to your own crate as a starting point.
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
//!   - Upstreams loaded from `aaagw.toml` with `match` glob patterns —
//!     requests are dispatched by inbound `model` field.
//!
//! Run:
//!
//! ```bash
//! # Required env vars:
//! #   TOKENS_FILE=tokens.toml
//! #   ADMIN_PASSWORD=<pwd>
//! #   USAGE_LOG=/var/log/aaagw/usage.jsonl
//! #   AAAGW_CONFIG=aaagw.toml   (upstream routing)
//! # Optional:
//! #   PRICING_CACHE=/etc/aaagw/pricing.json  (OpenRouter price book cache;
//! #                                           defaults to next to AAAGW_CONFIG)
//! cargo run --example 14_admin_relay \
//!     --features "server-anthropic,server-openai-chat,server-openai-responses,claude-code,codex" \
//!     -- 127.0.0.1:7878
//! ```

mod admin;
mod aggregate;
mod auth;
mod me;
mod pricing;
mod routes;
mod tokens;

use std::process::ExitCode;
use std::sync::Arc;

use agentix::server::{AnthropicServer, UsageLogger};

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
    let config_path = env_required("AAAGW_CONFIG");

    if tokens_path.is_none()
        || admin_password.is_none()
        || usage_log_path.is_none()
        || config_path.is_none()
    {
        eprintln!(
            "set env vars: TOKENS_FILE=tokens.toml ADMIN_PASSWORD=<pwd> \
             USAGE_LOG=/path.jsonl AAAGW_CONFIG=aaagw.toml"
        );
        return ExitCode::from(2);
    }
    let tokens_path = tokens_path.unwrap();
    let admin_password = admin_password.unwrap();
    let usage_log_path = usage_log_path.unwrap();
    let config_path = config_path.unwrap();

    let routes = match routes::RoutesHandle::load(&config_path) {
        Ok(r) => {
            tracing::info!(path = %config_path, count = r.len(), "routes loaded");
            r
        }
        Err(e) => {
            eprintln!("routes: {e}");
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

    // ── Price book (OpenRouter catalog) ───────────────────────────────
    // Optional cache path; defaults to `pricing.json` next to AAAGW_CONFIG.
    let pricing_cache = env_required("PRICING_CACHE").unwrap_or_else(|| {
        std::path::Path::new(&config_path)
            .parent()
            .map(|d| d.join("pricing.json"))
            .unwrap_or_else(|| std::path::PathBuf::from("pricing.json"))
            .to_string_lossy()
            .into_owned()
    });
    let pricing = pricing::PricingHandle::load(&pricing_cache).await;
    tracing::info!(path = %pricing_cache, models = pricing.len(), "price book ready");
    pricing.spawn_periodic_refresh();

    // ── Build the merged router ───────────────────────────────────────
    // Wrap the routes handle in `Arc<dyn ChainResolver>` and share across
    // all three protocol servers. Admin mutations to `routes` propagate
    // automatically on the next request.
    let resolver: Arc<dyn agentix::server::fallback::ChainResolver> = Arc::new(routes.clone());
    let mut router = axum::Router::new();

    let anthropic =
        AnthropicServer::with_resolver(resolver.clone()).with_usage_logger(usage_logger.clone());
    router = router.merge(anthropic.router());

    #[cfg(feature = "server-openai-chat")]
    {
        use agentix::server::OpenAIChatServer;
        let openai = OpenAIChatServer::with_resolver(resolver.clone())
            .with_usage_logger(usage_logger.clone());
        router = router.merge(openai.router());
    }

    #[cfg(feature = "server-openai-responses")]
    {
        use agentix::server::OpenAIResponsesServer;
        let resp = OpenAIResponsesServer::with_resolver(resolver.clone())
            .with_usage_logger(usage_logger.clone());
        router = router.merge(resp.router());
    }
    let _ = resolver;

    // Layers on /v1/* — outer is added LAST, so this stack is:
    //   request → token_auth → quota → handler
    let token_layer = auth::token_auth_layer(token_registry.clone());
    let quota = auth::quota_layer(
        token_registry.clone(),
        std::path::PathBuf::from(&usage_log_path),
    );
    router = router
        .layer(axum::middleware::from_fn(quota))
        .layer(axum::middleware::from_fn(token_layer));

    // /me routes — same token middleware applied internally by MeServer
    let me_server = me::MeServer::new(
        usage_log_path.clone(),
        token_registry.clone(),
        routes.clone(),
        pricing.clone(),
    );
    router = router.merge(me_server.router());

    // /admin routes (HTTP Basic).
    let admin_server = admin::AdminServer::new(
        usage_log_path.clone(),
        admin_password,
        token_registry,
        routes.clone(),
        pricing,
    );
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

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,agentix=info"));
    let _ = fmt().with_env_filter(filter).try_init();
}
