//! Admin dashboard + token registry + auth middlewares.
//!
//! Two routers and two middleware layers, intended to be merged into the same
//! axum app as the proxy readers (anthropic / openai_chat / openai_responses):
//!
//! - [`token_auth_layer`] gates `/v1/*` against [`TokenRegistry`]
//! - [`AdminServer`] mounts `/admin` + `/admin/api/*`, gated by HTTP Basic
//!
//! The dashboard reads the same `--usage-log` jsonl that the readers write,
//! so the data is always consistent without a separate ingestion step.

pub mod aggregate;
pub mod auth;
pub mod tokens;

use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::middleware;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;

pub use auth::{admin_basic_auth_layer, token_auth_layer};
pub use crate::server::usage::AuthedUser;
pub use tokens::{TokenEntry, TokenRegistry};

const DASHBOARD_HTML: &str = include_str!("dashboard.html");

#[derive(Clone)]
pub struct AdminServer {
    inner: Arc<Inner>,
}

struct Inner {
    usage_log_path: PathBuf,
    admin_password: String,
    tokens: TokenRegistry,
}

impl AdminServer {
    pub fn new(
        usage_log_path: impl Into<PathBuf>,
        admin_password: impl Into<String>,
        tokens: TokenRegistry,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                usage_log_path: usage_log_path.into(),
                admin_password: admin_password.into(),
                tokens,
            }),
        }
    }

    /// Build the `/admin` router, with HTTP Basic auth applied to every
    /// route below it.
    pub fn router(&self) -> Router {
        let basic = admin_basic_auth_layer(self.inner.admin_password.clone());
        Router::new()
            .route("/admin", get(dashboard_html))
            .route("/admin/", get(dashboard_html))
            .route("/admin/api/dashboard", get(dashboard_api))
            .route("/admin/api/tokens", get(tokens_api))
            .layer(middleware::from_fn(basic))
            .with_state(self.clone())
    }
}

async fn dashboard_html() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn dashboard_api(State(server): State<AdminServer>) -> Response {
    match aggregate::aggregate(&server.inner.usage_log_path, 100) {
        Ok(d) => Json(d).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "failed to read usage log");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("failed to read usage log: {e}"),
                })),
            )
                .into_response()
        }
    }
}

async fn tokens_api(State(server): State<AdminServer>) -> Response {
    let items: Vec<serde_json::Value> = server
        .inner
        .tokens
        .iter()
        .map(|(token, entry)| {
            serde_json::json!({
                "token": mask_token(token),
                "user": entry.user,
                "note": entry.note,
            })
        })
        .collect();
    Json(serde_json::json!({ "tokens": items })).into_response()
}

fn mask_token(t: &str) -> String {
    // Show first 8 chars + length, never expose the full secret over the
    // admin API. Even admins shouldn't be staring at raw secrets all day.
    let len = t.len();
    if len <= 8 {
        return format!("{}*", &t[..t.len().min(2)]);
    }
    format!("{}…({} chars)", &t[..8], len)
}

#[allow(dead_code)]
fn _silence_unused_imports(_: header::HeaderName) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_token_short_and_long() {
        assert_eq!(mask_token("sk-relay-abcdefghij"), "sk-relay…(19 chars)");
        assert_eq!(mask_token("abc"), "ab*");
    }
}
