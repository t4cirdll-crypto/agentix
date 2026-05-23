//! OpenAI Chat Completions-compatible HTTP server.
//!
//! Mounts `POST /v1/chat/completions` (streaming SSE + non-streaming JSON)
//! and `GET /v1/models`. Routes can be merged into the Anthropic server's
//! router for a single bind that speaks both formats simultaneously.

pub mod error;
pub mod inbound;
pub mod outbound;
pub mod wire;

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::stream::{self, BoxStream, Stream, StreamExt};
use serde_json::{Value, json};
use tokio::net::ToSocketAddrs;
use tracing::{error, info};

pub use error::{ErrorKind, OpenAIError};

use crate::server::fallback::{self, UpstreamSpec};

const MAX_REQUEST_BODY_BYTES: usize = 10 * 1024 * 1024;

#[derive(Clone)]
pub struct OpenAIChatServer {
    inner: Arc<Inner>,
}

struct Inner {
    chain: Vec<UpstreamSpec>,
    http: reqwest::Client,
    usage_logger: Option<Arc<crate::server::usage::UsageLogger>>,
}

impl OpenAIChatServer {
    pub fn new(chain: Vec<UpstreamSpec>) -> Self {
        Self::with_http_client(chain, reqwest::Client::new())
    }

    pub fn with_http_client(chain: Vec<UpstreamSpec>, http: reqwest::Client) -> Self {
        Self {
            inner: Arc::new(Inner {
                chain,
                http,
                usage_logger: None,
            }),
        }
    }

    pub fn with_usage_logger(self, logger: Arc<crate::server::usage::UsageLogger>) -> Self {
        Self {
            inner: Arc::new(Inner {
                chain: self.inner.chain.clone(),
                http: self.inner.http.clone(),
                usage_logger: Some(logger),
            }),
        }
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/models", get(handle_models))
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
            .with_state(self.clone())
    }

    pub async fn listen<A: ToSocketAddrs>(self, addr: A) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        info!(%local, "agentix openai-chat server listening");
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown_signal())
            .await
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

// ── Handlers ────────────────────────────────────────────────────────────────

async fn handle_chat_completions(
    State(server): State<OpenAIChatServer>,
    headers: axum::http::HeaderMap,
    authed: Option<axum::Extension<crate::server::usage::AuthedUser>>,
    Json(body): Json<wire::ChatCompletionsRequest>,
) -> Response {
    let request_model = body.model.clone();
    let stream_requested = body.stream.unwrap_or(false);
    let include_usage = body
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let auth_token = crate::server::usage::extract_client_token(&headers);
    let resolved_user = authed.map(|axum::Extension(u)| u.user);
    let mut tracker = crate::server::usage::UsageTracker::new(
        server.inner.usage_logger.clone(),
        "openai_chat",
        request_model.clone(),
        auth_token,
        stream_requested,
    );
    tracker.set_user(resolved_user);

    let translated = match inbound::translate(body) {
        Ok(t) => t,
        Err(e) => {
            tracker.mark_error(format!("{e}"));
            tracker.finalize();
            return e.into_response();
        }
    };

    if stream_requested {
        let chain = server.inner.chain.clone();
        let http = server.inner.http.clone();
        match fallback::stream_with_fallback(chain, translated, http).await {
            Ok((llm_stream, committed)) => {
                tracker.set_committed(committed);
                sse_response(llm_stream, request_model, include_usage, tracker)
            }
            Err(e) => {
                error!(error = %e, "all upstreams failed before commit");
                tracker.mark_error(format!("{e}"));
                tracker.finalize();
                OpenAIError::server(format!("all upstreams failed: {e}")).into_response()
            }
        }
    } else {
        match fallback::complete_with_fallback(&server.inner.chain, &translated, &server.inner.http)
            .await
        {
            Ok((resp, committed)) => {
                tracker.set_committed(committed);
                tracker.set_usage(resp.usage.clone());
                let body = outbound::build_response_body(resp, &request_model);
                tracker.finalize();
                Json(body).into_response()
            }
            Err(e) => {
                error!(error = %e, "all upstreams failed");
                tracker.mark_error(format!("{e}"));
                tracker.finalize();
                OpenAIError::server(format!("all upstreams failed: {e}")).into_response()
            }
        }
    }
}

fn sse_response(
    llm_stream: BoxStream<'static, crate::msg::LlmEvent>,
    model: String,
    include_usage: bool,
    tracker: crate::server::usage::UsageTracker,
) -> Response {
    let state = outbound::ChunkState::new(model, include_usage);
    let event_stream = sse_events(state, llm_stream, tracker);
    Sse::new(event_stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

fn sse_events(
    state: outbound::ChunkState,
    llm_stream: BoxStream<'static, crate::msg::LlmEvent>,
    tracker: crate::server::usage::UsageTracker,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    stream::unfold(
        (
            state,
            llm_stream,
            std::collections::VecDeque::<(&'static str, Value)>::new(),
            false,
            Some(tracker),
        ),
        |(mut state, mut stream, mut buffered, mut finished, mut tracker)| async move {
            loop {
                if let Some((_, payload)) = buffered.pop_front() {
                    // Chat Completions SSE has no `event:` line — every frame
                    // is plain `data: <json>` (or `data: [DONE]`).
                    let body = match &payload {
                        Value::String(s) => s.clone(),
                        v => v.to_string(),
                    };
                    let event = Event::default().data(body);
                    return Some((
                        Ok::<_, Infallible>(event),
                        (state, stream, buffered, finished, tracker),
                    ));
                }
                if finished {
                    if let Some(t) = tracker.take() {
                        t.finalize();
                    }
                    return None;
                }
                match stream.next().await {
                    Some(ev) => {
                        if let Some(t) = tracker.as_mut() {
                            if t.observe(&ev) {
                                finished = true;
                            }
                        } else if matches!(
                            ev,
                            crate::msg::LlmEvent::Done | crate::msg::LlmEvent::Error(_)
                        ) {
                            finished = true;
                        }
                        for frame in state.on_event(ev) {
                            buffered.push_back(frame);
                        }
                    }
                    None => {
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

async fn handle_models(State(server): State<OpenAIChatServer>) -> Response {
    let mut data: Vec<Value> = Vec::new();
    let now = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    };
    for spec in server.inner.chain.iter() {
        let id = spec
            .model
            .clone()
            .unwrap_or_else(|| spec.provider.default_model().to_string());
        data.push(json!({
            "id": id,
            "object": "model",
            "created": now,
            "owned_by": format!("{:?}", spec.provider).to_lowercase(),
        }));
    }
    Json(json!({"object": "list", "data": data})).into_response()
}
