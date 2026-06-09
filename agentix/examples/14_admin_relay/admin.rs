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
//! Admins can also mutate tokens and routes through the API — changes
//! persist to disk and take effect immediately (no service restart).

use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::middleware;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use serde::Deserialize;

pub use crate::auth::{admin_basic_auth_layer, token_auth_layer};
pub use crate::tokens::{TokenEntry, TokenRegistry};

use crate::pricing::PricingHandle;
use crate::routes::{Route, RoutesHandle};

const DASHBOARD_HTML: &str = include_str!("dashboard.html");

#[derive(Clone)]
pub struct AdminServer {
    inner: Arc<Inner>,
}

struct Inner {
    usage_log_path: PathBuf,
    admin_password: String,
    tokens: TokenRegistry,
    routes: RoutesHandle,
    pricing: PricingHandle,
}

impl AdminServer {
    pub fn new(
        usage_log_path: impl Into<PathBuf>,
        admin_password: impl Into<String>,
        tokens: TokenRegistry,
        routes: RoutesHandle,
        pricing: PricingHandle,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                usage_log_path: usage_log_path.into(),
                admin_password: admin_password.into(),
                tokens,
                routes,
                pricing,
            }),
        }
    }

    pub fn router(&self) -> Router {
        let basic = admin_basic_auth_layer(self.inner.admin_password.clone());
        Router::new()
            .route("/admin", get(dashboard_html))
            .route("/admin/", get(dashboard_html))
            .route("/admin/api/dashboard", get(dashboard_api))
            .route("/admin/api/tokens", get(tokens_list).post(tokens_create))
            .route(
                "/admin/api/tokens/{token}",
                delete(tokens_revoke).put(tokens_update),
            )
            .route(
                "/admin/api/routes",
                get(routes_list).post(routes_create).put(routes_replace_all),
            )
            .route(
                "/admin/api/routes/{index}",
                put(routes_update).delete(routes_remove),
            )
            .layer(middleware::from_fn(basic))
            .with_state(self.clone())
    }
}

async fn dashboard_html() -> impl IntoResponse {
    // No-cache so browsers always pick up a freshly-deployed dashboard rather
    // than serving a heuristically-cached older copy.
    (
        [(
            axum::http::header::CACHE_CONTROL,
            "no-cache, must-revalidate",
        )],
        Html(DASHBOARD_HTML),
    )
}

async fn dashboard_api(State(server): State<AdminServer>) -> Response {
    let pricer =
        crate::pricing::record_pricer(server.inner.pricing.clone(), server.inner.routes.clone());
    match crate::aggregate::aggregate(&server.inner.usage_log_path, 100, &pricer) {
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

// ── Tokens ──────────────────────────────────────────────────────────────

async fn tokens_list(State(server): State<AdminServer>) -> Response {
    let items: Vec<serde_json::Value> = server
        .inner
        .tokens
        .snapshot()
        .into_iter()
        .map(|(token, entry)| {
            serde_json::json!({
                "token": token,                 // full token — admin needs to copy when minting
                "token_masked": mask_token(&token),
                "user": entry.user,
                "note": entry.note,
                "monthly_token_budget": entry.monthly_token_budget,
            })
        })
        .collect();
    Json(serde_json::json!({ "tokens": items })).into_response()
}

#[derive(Deserialize)]
struct NewTokenRequest {
    user: String,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    monthly_token_budget: Option<u64>,
    /// Optional explicit token. When omitted the server mints a random one.
    #[serde(default)]
    token: Option<String>,
}

async fn tokens_create(
    State(server): State<AdminServer>,
    Json(req): Json<NewTokenRequest>,
) -> Response {
    if req.user.trim().is_empty() {
        return bad_request("user is required");
    }
    let token = req.token.unwrap_or_else(generate_token);
    let entry = TokenEntry {
        user: req.user,
        note: req.note,
        monthly_token_budget: req.monthly_token_budget,
    };
    match server.inner.tokens.upsert(token.clone(), entry.clone()) {
        Ok(()) => Json(serde_json::json!({
            "token": token,
            "user": entry.user,
            "note": entry.note,
            "monthly_token_budget": entry.monthly_token_budget,
        }))
        .into_response(),
        Err(e) => internal_error(e),
    }
}

#[derive(Deserialize)]
struct UpdateTokenRequest {
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    note: Option<Option<String>>,
    #[serde(default)]
    monthly_token_budget: Option<Option<u64>>,
}

async fn tokens_update(
    State(server): State<AdminServer>,
    AxPath(token): AxPath<String>,
    Json(req): Json<UpdateTokenRequest>,
) -> Response {
    let Some(mut entry) = server.inner.tokens.lookup(&token) else {
        return not_found("token not found");
    };
    if let Some(u) = req.user {
        if u.trim().is_empty() {
            return bad_request("user cannot be empty");
        }
        entry.user = u;
    }
    if let Some(n) = req.note {
        entry.note = n;
    }
    if let Some(b) = req.monthly_token_budget {
        entry.monthly_token_budget = b;
    }
    match server.inner.tokens.upsert(token.clone(), entry.clone()) {
        Ok(()) => Json(serde_json::json!({
            "token": token,
            "user": entry.user,
            "note": entry.note,
            "monthly_token_budget": entry.monthly_token_budget,
        }))
        .into_response(),
        Err(e) => internal_error(e),
    }
}

async fn tokens_revoke(
    State(server): State<AdminServer>,
    AxPath(token): AxPath<String>,
) -> Response {
    match server.inner.tokens.remove(&token) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => not_found("token not found"),
        Err(e) => internal_error(e),
    }
}

// ── Routes ──────────────────────────────────────────────────────────────

async fn routes_list(State(server): State<AdminServer>) -> Response {
    let items: Vec<serde_json::Value> = server
        .inner
        .routes
        .snapshot()
        .into_iter()
        .map(|r| serde_json::to_value(&r).unwrap_or(serde_json::Value::Null))
        .collect();
    Json(serde_json::json!({ "routes": items })).into_response()
}

async fn routes_create(State(server): State<AdminServer>, Json(route): Json<Route>) -> Response {
    if let Err(e) = validate_route(&route) {
        return bad_request(e);
    }
    match server.inner.routes.append(route.clone()) {
        Ok(idx) => Json(serde_json::json!({
            "index": idx,
            "route": route,
        }))
        .into_response(),
        Err(e) => internal_error(e),
    }
}

async fn routes_update(
    State(server): State<AdminServer>,
    AxPath(index): AxPath<usize>,
    Json(route): Json<Route>,
) -> Response {
    if let Err(e) = validate_route(&route) {
        return bad_request(e);
    }
    match server.inner.routes.update(index, route.clone()) {
        Ok(()) => Json(serde_json::json!({
            "index": index,
            "route": route,
        }))
        .into_response(),
        Err(e) => bad_request(e),
    }
}

async fn routes_remove(
    State(server): State<AdminServer>,
    AxPath(index): AxPath<usize>,
) -> Response {
    match server.inner.routes.remove(index) {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => bad_request(e),
    }
}

#[derive(Deserialize)]
struct ReplaceAllRoutes {
    routes: Vec<Route>,
}

async fn routes_replace_all(
    State(server): State<AdminServer>,
    Json(body): Json<ReplaceAllRoutes>,
) -> Response {
    for r in &body.routes {
        if let Err(e) = validate_route(r) {
            return bad_request(e);
        }
    }
    match server.inner.routes.replace_all(body.routes.clone()) {
        Ok(_) => Json(serde_json::json!({ "count": body.routes.len() })).into_response(),
        Err(e) => internal_error(e),
    }
}

fn validate_route(r: &Route) -> Result<(), String> {
    if r.match_pattern.trim().is_empty() {
        return Err("match pattern cannot be empty".into());
    }
    if r.fallback.is_empty() {
        return Err("route must have at least one fallback entry".into());
    }
    for u in &r.fallback {
        if u.target.trim().is_empty() {
            return Err("upstream target cannot be empty".into());
        }
    }
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn mask_token(t: &str) -> String {
    let len = t.len();
    if len <= 8 {
        return format!("{}*", &t[..t.len().min(2)]);
    }
    format!("{}…({} chars)", &t[..8], len)
}

fn generate_token() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Mash together nanos + a few extra entropy bytes for the example;
    // production deployments should use a CSPRNG.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let salt: u64 = ((std::ptr::addr_of!(nanos) as usize as u64) ^ nanos as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    format!("sk-relay-{:016x}{:016x}", nanos, salt)
}

fn bad_request(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}

fn not_found(msg: impl Into<String>) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}

fn internal_error(msg: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": msg.into()})),
    )
        .into_response()
}
