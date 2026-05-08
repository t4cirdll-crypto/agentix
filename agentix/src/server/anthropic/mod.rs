//! Anthropic Messages-compatible HTTP server.
//!
//! Speaks the Anthropic Messages wire format on inbound and forwards each
//! request to a chain of agentix upstreams with fallback. Drop-in proxy for
//! tools that hardcode Anthropic's API shape.
//!
//! ```no_run
//! # use agentix::{Provider};
//! # use agentix::server::{AnthropicServer, UpstreamSpec};
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let chain = vec![
//!     UpstreamSpec::new(Provider::Anthropic, std::env::var("ANTHROPIC_API_KEY")?),
//! ];
//! AnthropicServer::new(chain).listen("127.0.0.1:7878").await?;
//! # Ok(()) }
//! ```

pub mod error;
pub mod inbound;
pub mod outbound;

use crate::server::fallback;

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
#[cfg(not(feature = "server-openai-chat"))]
use axum::routing::get;
use axum::routing::post;
use futures::stream::{self, BoxStream, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::net::ToSocketAddrs;
use tracing::{error, info};

pub use error::{ErrorKind, ServerError};
pub use crate::server::fallback::UpstreamSpec;

const MAX_REQUEST_BODY_BYTES: usize = 10 * 1024 * 1024;

/// HTTP server exposing an Anthropic Messages-compatible endpoint backed by
/// a fallback chain of agentix upstreams.
#[derive(Clone)]
pub struct AnthropicServer {
    inner: Arc<Inner>,
}

struct Inner {
    chain: Vec<UpstreamSpec>,
    http: reqwest::Client,
}

impl AnthropicServer {
    /// Create a new server with the given fallback chain. Order matters: the
    /// first upstream is tried first; on any error before its stream commits
    /// to its first event, the next upstream is tried.
    pub fn new(chain: Vec<UpstreamSpec>) -> Self {
        Self::with_http_client(chain, reqwest::Client::new())
    }

    /// Same as [`AnthropicServer::new`] but uses a caller-provided
    /// `reqwest::Client` for upstream HTTP traffic.
    pub fn with_http_client(chain: Vec<UpstreamSpec>, http: reqwest::Client) -> Self {
        Self {
            inner: Arc::new(Inner { chain, http }),
        }
    }

    /// Build the axum router. Useful when embedding the server inside another
    /// axum application.
    pub fn router(&self) -> Router {
        #[cfg_attr(feature = "server-openai-chat", allow(unused_mut))]
        let mut r = Router::new()
            .route("/v1/messages", post(handle_messages))
            .route("/v1/messages/count_tokens", post(handle_count_tokens));
        // Only mount /v1/models when standing alone — when merged with the
        // OpenAI Chat router it owns this path.
        #[cfg(not(feature = "server-openai-chat"))]
        {
            r = r.route("/v1/models", get(handle_models));
        }
        r.fallback(handle_fallback)
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
            .with_state(self.clone())
    }

    /// Bind to `addr` and serve until Ctrl-C is received.
    pub async fn listen<A: ToSocketAddrs>(self, addr: A) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        info!(%local, "agentix anthropic-messages server listening");
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown_signal())
            .await
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}

// ── Handlers ────────────────────────────────────────────────────────────────

async fn handle_messages(
    State(server): State<AnthropicServer>,
    Json(body): Json<inbound::IncomingRequest>,
) -> Response {
    let stream_requested = body.stream.unwrap_or(false);
    let request_model = body.model.clone();

    let translated = match inbound::translate(body) {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };

    if stream_requested {
        let chain = server.inner.chain.clone();
        let http = server.inner.http.clone();
        match fallback::stream_with_fallback(chain, translated, http).await {
            Ok(llm_stream) => sse_response(llm_stream, request_model),
            Err(e) => {
                error!(error = %e, "all upstreams failed before commit");
                ServerError::api(format!("all upstreams failed: {e}")).into_response()
            }
        }
    } else {
        match fallback::complete_with_fallback(
            &server.inner.chain,
            &translated,
            &server.inner.http,
        )
        .await
        {
            Ok(resp) => {
                let has_reasoning = translated.reasoning_effort.is_some()
                    && translated.reasoning_effort != Some(crate::request::ReasoningEffort::None);
                let body = outbound::build_response_body(resp, &request_model, has_reasoning);
                Json(body).into_response()
            }
            Err(e) => {
                error!(error = %e, "all upstreams failed");
                ServerError::api(format!("all upstreams failed: {e}")).into_response()
            }
        }
    }
}

fn sse_response(llm_stream: BoxStream<'static, crate::msg::LlmEvent>, model: String) -> Response {
    let state = outbound::SseState::new(model);
    let event_stream = sse_events(state, llm_stream);
    Sse::new(event_stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

fn sse_events(
    state: outbound::SseState,
    llm_stream: BoxStream<'static, crate::msg::LlmEvent>,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    stream::unfold(
        (state, llm_stream, std::collections::VecDeque::<(&'static str, Value)>::new(), false),
        |(mut state, mut stream, mut buffered, mut finished)| async move {
            loop {
                if let Some((name, payload)) = buffered.pop_front() {
                    let event = Event::default()
                        .event(name)
                        .data(payload.to_string());
                    return Some((Ok::<_, Infallible>(event), (state, stream, buffered, finished)));
                }
                if finished {
                    return None;
                }
                match stream.next().await {
                    Some(ev) => {
                        let is_done = matches!(ev, crate::msg::LlmEvent::Done);
                        let is_error = matches!(ev, crate::msg::LlmEvent::Error(_));
                        for frame in state.on_event(ev) {
                            buffered.push_back(frame);
                        }
                        if is_done || is_error {
                            finished = true;
                        }
                    }
                    None => {
                        // Stream ended without Done. Synthesize a Done so the
                        // wire output is well-formed.
                        for frame in state.on_event(crate::msg::LlmEvent::Done) {
                            buffered.push_back(frame);
                        }
                        finished = true;
                    }
                }
            }
        },
    )
}

async fn handle_count_tokens(Json(body): Json<inbound::IncomingRequest>) -> Response {
    let translated = match inbound::translate(body) {
        Ok(t) => t,
        Err(e) => return e.into_response(),
    };
    let mut total: usize = translated
        .messages
        .iter()
        .map(|m| m.estimate_tokens())
        .sum();
    if let Some(sys) = &translated.system_prompt {
        total += crate::request::Message::User(vec![crate::request::Content::text(sys.clone())])
            .estimate_tokens();
    }
    Json(json!({"input_tokens": total})).into_response()
}

#[cfg_attr(feature = "server-openai-chat", allow(dead_code))]
async fn handle_models(State(server): State<AnthropicServer>) -> Response {
    let mut data: Vec<Value> = Vec::new();
    for (i, spec) in server.inner.chain.iter().enumerate() {
        let id = spec
            .model
            .clone()
            .unwrap_or_else(|| spec.provider.default_model().to_string());
        data.push(json!({
            "type": "model",
            "id": id,
            "display_name": id,
            "rank": i,
        }));
    }
    Json(json!({"data": data, "has_more": false, "first_id": Value::Null, "last_id": Value::Null}))
        .into_response()
}

async fn handle_fallback() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(error::error_body(
            ErrorKind::NotFound,
            "endpoint not implemented by agentix anthropic server",
        )),
    )
        .into_response()
}
