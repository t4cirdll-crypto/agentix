//! Bearer-token / x-api-key auth middleware for the proxy endpoints.
//!
//! Two layers:
//!
//! 1. [`token_auth_layer`] gates `/v1/*` routes. Looks up the inbound
//!    `Authorization: Bearer <t>` (OpenAI-style) or `x-api-key: <t>`
//!    (Anthropic-style) header against a [`TokenRegistry`]; 401 on miss.
//!    On match, attaches the user name to request extensions so the usage
//!    logger can pull it out.
//!
//! 2. [`admin_basic_auth_layer`] gates `/admin/*` routes via HTTP Basic
//!    against a fixed admin password (no username check). 401 + WWW-
//!    Authenticate on miss.

use axum::extract::Request;
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use base64::Engine;
use serde_json::json;

use super::tokens::TokenRegistry;
use crate::server::usage::AuthedUser;

/// Build a middleware function that validates the incoming token against
/// the registry. Cloneable; the registry is `Arc`-backed.
pub fn token_auth_layer(
    registry: TokenRegistry,
) -> impl Clone
+ Send
+ Sync
+ 'static
+ Fn(
    Request,
    Next,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> {
    move |mut req: Request, next: Next| {
        let registry = registry.clone();
        Box::pin(async move {
            let token = extract_proxy_token(req.headers());
            let entry = token.as_deref().and_then(|t| registry.lookup(t));
            let Some(entry) = entry else {
                return unauthorized_json("missing or unknown API key");
            };
            req.extensions_mut().insert(AuthedUser {
                token: token.unwrap_or_default(),
                user: entry.user.clone(),
            });
            next.run(req).await
        })
    }
}

fn extract_proxy_token(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(auth) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        let trimmed = auth.trim();
        let bare = trimmed
            .strip_prefix("Bearer ")
            .or_else(|| trimmed.strip_prefix("bearer "))
            .unwrap_or(trimmed);
        if !bare.is_empty() {
            return Some(bare.to_string());
        }
    }
    let xapi = headers.get("x-api-key").and_then(|v| v.to_str().ok())?;
    let t = xapi.trim();
    if t.is_empty() { None } else { Some(t.to_string()) }
}

fn unauthorized_json(message: &str) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": "authentication_error",
            "param": serde_json::Value::Null,
            "code": serde_json::Value::Null,
        }
    });
    (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
}

/// Build a middleware that requires `Authorization: Basic ...` matching
/// `admin_password`. Username is not checked.
pub fn admin_basic_auth_layer(
    admin_password: String,
) -> impl Clone
+ Send
+ Sync
+ 'static
+ Fn(
    Request,
    Next,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>> {
    move |req: Request, next: Next| {
        let admin_password = admin_password.clone();
        Box::pin(async move {
            let provided = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().strip_prefix("Basic "))
                .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok())
                .and_then(|bytes| String::from_utf8(bytes).ok());

            let ok = match provided {
                Some(creds) => {
                    let (_user, pwd) = creds.split_once(':').unwrap_or(("", &creds));
                    pwd == admin_password
                }
                None => false,
            };

            if !ok {
                let mut resp = (
                    StatusCode::UNAUTHORIZED,
                    "agentix admin: authentication required",
                )
                    .into_response();
                resp.headers_mut().insert(
                    header::WWW_AUTHENTICATE,
                    HeaderValue::from_static("Basic realm=\"agentix admin\""),
                );
                return resp;
            }
            next.run(req).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn extracts_bearer_token() {
        let mut h = axum::http::HeaderMap::new();
        h.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-x"),
        );
        assert_eq!(extract_proxy_token(&h).as_deref(), Some("sk-x"));
    }

    #[test]
    fn extracts_x_api_key() {
        let mut h = axum::http::HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("sk-x"));
        assert_eq!(extract_proxy_token(&h).as_deref(), Some("sk-x"));
    }

    #[test]
    fn empty_returns_none() {
        let h = axum::http::HeaderMap::new();
        assert!(extract_proxy_token(&h).is_none());
    }
}
