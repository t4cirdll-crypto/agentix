//! Per-user self-service endpoints under `/me`. Authentication uses the
//! same bearer-token middleware as `/v1/*` — users hit `/me` with their
//! relay API key and get back only their own slice of the usage log.

use std::path::PathBuf;
use std::sync::Arc;

use agentix::server::AuthedUser;
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::middleware;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;

use crate::auth;
use crate::pricing::PricingHandle;
use crate::routes::RoutesHandle;
use crate::tokens::TokenRegistry;

const ME_HTML: &str = include_str!("me.html");

#[derive(Clone)]
pub struct MeServer {
    inner: Arc<Inner>,
}

struct Inner {
    usage_log_path: PathBuf,
    tokens: TokenRegistry,
    routes: RoutesHandle,
    pricing: PricingHandle,
}

impl MeServer {
    pub fn new(
        usage_log_path: impl Into<PathBuf>,
        tokens: TokenRegistry,
        routes: RoutesHandle,
        pricing: PricingHandle,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                usage_log_path: usage_log_path.into(),
                tokens,
                routes,
                pricing,
            }),
        }
    }

    /// Router under `/me` with token-auth applied. The same TokenRegistry
    /// the proxy uses gates these routes.
    pub fn router(&self) -> Router {
        let auth = auth::token_auth_layer(self.inner.tokens.clone());
        Router::new()
            .route("/me", get(me_html))
            .route("/me/api/usage", get(me_usage))
            .layer(middleware::from_fn(auth))
            .with_state(self.clone())
    }
}

async fn me_html() -> Html<&'static str> {
    Html(ME_HTML)
}

async fn me_usage(
    State(server): State<MeServer>,
    authed: Option<axum::Extension<AuthedUser>>,
) -> Response {
    let Some(axum::Extension(user)) = authed else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "missing API key"}})),
        )
            .into_response();
    };

    let budget = server
        .inner
        .tokens
        .lookup(&user.token)
        .and_then(|e| e.monthly_token_budget);

    let pricer =
        crate::pricing::record_pricer(server.inner.pricing.clone(), server.inner.routes.clone());
    match crate::aggregate::user_month_summary(
        &server.inner.usage_log_path,
        &user.user,
        50,
        budget,
        &pricer,
    ) {
        Ok(s) => Json(s).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("failed to read usage log: {e}")
            })),
        )
            .into_response(),
    }
}
