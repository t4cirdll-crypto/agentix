//! OpenAI Responses API-compatible HTTP server.
//!
//! Mounts `POST /v1/responses` (streaming SSE + non-streaming JSON). Maintains
//! an in-memory session store so `previous_response_id` chaining works for
//! clients that default to `store: true` (notably Codex CLI).
//!
//! Built-in tools (web_search / file_search / computer_use / code_interpreter)
//! are rejected at translation time — agentix has no equivalent in the AST.
//! Custom function tools work normally.

pub mod error;
pub mod inbound;
pub mod outbound;
pub mod store;
pub mod wire;

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use futures::stream::{self, BoxStream, Stream, StreamExt};
use serde_json::Value;
use tokio::net::ToSocketAddrs;
use tracing::{error, info};

pub use error::{ErrorKind, OpenAIError};
pub use store::SessionStore;

use crate::server::fallback::{self, UpstreamSpec};

const MAX_REQUEST_BODY_BYTES: usize = 10 * 1024 * 1024;

#[derive(Clone)]
pub struct OpenAIResponsesServer {
    inner: Arc<Inner>,
}

struct Inner {
    resolver: Arc<dyn crate::server::fallback::ChainResolver>,
    http: reqwest::Client,
    store: Arc<SessionStore>,
    /// When true, never persist session items and reject any request that
    /// carries `previous_response_id`. Useful for multi-replica deployments
    /// (where in-memory state can't be shared) and for environments that
    /// must not retain conversation data.
    stateless: bool,
    usage_logger: Option<Arc<crate::server::usage::UsageLogger>>,
}

impl OpenAIResponsesServer {
    pub fn new(chain: Vec<UpstreamSpec>) -> Self {
        Self::with_options(
            chain,
            reqwest::Client::new(),
            Arc::new(SessionStore::default()),
        )
    }

    pub fn with_options(
        chain: Vec<UpstreamSpec>,
        http: reqwest::Client,
        store: Arc<SessionStore>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                resolver: Arc::new(chain),
                http,
                store,
                stateless: false,
                usage_logger: None,
            }),
        }
    }

    pub fn with_resolver(resolver: Arc<dyn crate::server::fallback::ChainResolver>) -> Self {
        Self {
            inner: Arc::new(Inner {
                resolver,
                http: reqwest::Client::new(),
                store: Arc::new(SessionStore::default()),
                stateless: false,
                usage_logger: None,
            }),
        }
    }

    /// Disable session storage. Even when clients send `store: true`, the
    /// proxy never persists; clients sending `previous_response_id` are
    /// rejected with `invalid_request_error`. Use for horizontal scaling
    /// and zero-retention deployments.
    pub fn stateless(self) -> Self {
        Self {
            inner: Arc::new(Inner {
                resolver: self.inner.resolver.clone(),
                http: self.inner.http.clone(),
                store: self.inner.store.clone(),
                stateless: true,
                usage_logger: self.inner.usage_logger.clone(),
            }),
        }
    }

    pub fn with_usage_logger(self, logger: Arc<crate::server::usage::UsageLogger>) -> Self {
        Self {
            inner: Arc::new(Inner {
                resolver: self.inner.resolver.clone(),
                http: self.inner.http.clone(),
                store: self.inner.store.clone(),
                stateless: self.inner.stateless,
                usage_logger: Some(logger),
            }),
        }
    }

    pub fn is_stateless(&self) -> bool {
        self.inner.stateless
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/v1/responses", post(handle_responses))
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
            .with_state(self.clone())
    }

    pub async fn listen<A: ToSocketAddrs>(self, addr: A) -> std::io::Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local = listener.local_addr()?;
        info!(%local, "agentix openai-responses server listening");
        axum::serve(listener, self.router())
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
    }
}

// ── Handlers ────────────────────────────────────────────────────────────────

async fn handle_responses(
    State(server): State<OpenAIResponsesServer>,
    headers: axum::http::HeaderMap,
    authed: Option<axum::Extension<crate::server::usage::AuthedUser>>,
    Json(body): Json<wire::ResponsesRequest>,
) -> Response {
    let request_model = body.model.clone();
    let stateless = server.inner.stateless;
    let stream_requested_early = body.stream.unwrap_or(false);
    let auth_token = crate::server::usage::extract_client_token(&headers);
    let resolved_user = authed.map(|axum::Extension(u)| u.user);
    let mut tracker = crate::server::usage::UsageTracker::new(
        server.inner.usage_logger.clone(),
        "openai_responses",
        request_model.clone(),
        auth_token,
        stream_requested_early,
    );
    tracker.set_user(resolved_user);

    if stateless && body.previous_response_id.is_some() {
        let err = OpenAIError::invalid_request(
            "this proxy is running in stateless mode; previous_response_id is not supported. \
             Send the full conversation history in `input` each turn.",
        );
        tracker.mark_error(format!("{err}"));
        tracker.finalize();
        return err.into_response();
    }

    let prepared = match inbound::translate(
        body,
        &inbound::InboundContext {
            store: server.inner.store.clone(),
        },
    ) {
        Ok(p) => p,
        Err(e) => {
            tracker.mark_error(format!("{e}"));
            tracker.finalize();
            return e.into_response();
        }
    };

    let response_id = outbound::synth_response_id();
    let stream_requested = prepared.translated.stream;
    let parent_id = prepared.parent_id.clone();
    let stored_input = prepared.stored_input_items.clone();
    let store_requested = prepared.store_requested && !stateless;
    let reasoning_summary = prepared.reasoning_summary.clone();
    let instructions = prepared.translated.system_prompt.clone();
    let store = server.inner.store.clone();

    if stream_requested {
        let chain = server.inner.resolver.resolve(&request_model);
        let http = server.inner.http.clone();
        match fallback::stream_with_fallback(chain, prepared.translated, http).await {
            Ok((llm_stream, committed)) => {
                tracker.set_committed(committed);
                sse_response(
                    llm_stream,
                    response_id,
                    request_model,
                    instructions,
                    parent_id.clone(),
                    reasoning_summary,
                    store_requested,
                    store,
                    stored_input,
                    tracker,
                )
            }
            Err(e) => {
                error!(error = %e, "all upstreams failed before commit");
                tracker.mark_error(format!("{e}"));
                tracker.finalize();
                OpenAIError::server(format!("all upstreams failed: {e}")).into_response()
            }
        }
    } else {
        let chain = server.inner.resolver.resolve(&request_model);
        match fallback::complete_with_fallback(&chain, &prepared.translated, &server.inner.http)
            .await
        {
            Ok((resp, committed)) => {
                tracker.set_committed(committed);
                tracker.set_usage(resp.usage.clone());
                // Snapshot the items we'll persist.
                let mut all_items = stored_input;
                if let Some(pd) = resp
                    .provider_data
                    .as_ref()
                    .and_then(|v| v.get("openai_responses_items"))
                    .and_then(|v| v.as_array())
                {
                    all_items.extend(pd.iter().cloned());
                } else {
                    if let Some(reasoning) = resp.reasoning.as_deref().filter(|s| !s.is_empty()) {
                        all_items.push(serde_json::json!({
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": reasoning}],
                        }));
                    }
                    if let Some(text) = resp.content.as_deref().filter(|t| !t.is_empty()) {
                        all_items.push(serde_json::json!({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type":"output_text","text":text,"annotations":[]}],
                        }));
                    }
                    for tc in &resp.tool_calls {
                        all_items.push(serde_json::json!({
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }));
                    }
                }
                if store_requested {
                    store.put(response_id.clone(), all_items, parent_id.clone());
                }
                let body = outbound::build_non_streaming(
                    resp,
                    &request_model,
                    parent_id.as_deref(),
                    &response_id,
                    instructions,
                    reasoning_summary,
                );
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

#[allow(clippy::too_many_arguments)]
fn sse_response(
    llm_stream: BoxStream<'static, crate::msg::LlmEvent>,
    response_id: String,
    model: String,
    instructions: Option<String>,
    parent_id: Option<String>,
    reasoning_summary: Option<String>,
    store_requested: bool,
    store: Arc<SessionStore>,
    stored_input_items: Vec<Value>,
    tracker: crate::server::usage::UsageTracker,
) -> Response {
    let state = outbound::ResponsesStreamState::new(
        response_id.clone(),
        model,
        instructions,
        parent_id.clone(),
        reasoning_summary,
    );
    let event_stream = sse_events(
        state,
        llm_stream,
        store,
        store_requested,
        response_id,
        parent_id,
        stored_input_items,
        tracker,
    );
    Sse::new(event_stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

#[allow(clippy::too_many_arguments)]
fn sse_events(
    state: outbound::ResponsesStreamState,
    llm_stream: BoxStream<'static, crate::msg::LlmEvent>,
    store: Arc<SessionStore>,
    store_requested: bool,
    response_id: String,
    parent_id: Option<String>,
    stored_input_items: Vec<Value>,
    tracker: crate::server::usage::UsageTracker,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    stream::unfold(
        (
            state,
            llm_stream,
            std::collections::VecDeque::<(&'static str, Value)>::new(),
            false,
            false, // saw_completed
            store,
            store_requested,
            response_id,
            parent_id,
            stored_input_items,
            Some(tracker),
        ),
        |(
            mut state,
            mut stream,
            mut buffered,
            mut finished,
            mut saw_completed,
            store,
            store_requested,
            response_id,
            parent_id,
            stored_input_items,
            mut tracker,
        )| async move {
            loop {
                if let Some((name, payload)) = buffered.pop_front() {
                    let event = Event::default().event(name).data(payload.to_string());
                    return Some((
                        Ok::<_, Infallible>(event),
                        (
                            state,
                            stream,
                            buffered,
                            finished,
                            saw_completed,
                            store,
                            store_requested,
                            response_id,
                            parent_id,
                            stored_input_items,
                            tracker,
                        ),
                    ));
                }
                if finished {
                    // Persist on successful completion.
                    if saw_completed && store_requested {
                        let mut items = stored_input_items.clone();
                        items.extend(state.committed_items.iter().cloned());
                        store.put(response_id.clone(), items, parent_id.clone());
                    }
                    if let Some(t) = tracker.take() {
                        t.finalize();
                    }
                    return None;
                }
                match stream.next().await {
                    Some(ev) => {
                        let is_done = matches!(ev, crate::msg::LlmEvent::Done);
                        let is_error = matches!(ev, crate::msg::LlmEvent::Error(_));
                        if let Some(t) = tracker.as_mut() {
                            t.observe(&ev);
                        }
                        for frame in state.on_event(ev) {
                            buffered.push_back(frame);
                        }
                        if is_done {
                            saw_completed = true;
                            finished = true;
                        }
                        if is_error {
                            finished = true;
                        }
                    }
                    None => {
                        for frame in state.on_event(crate::msg::LlmEvent::Done) {
                            buffered.push_back(frame);
                        }
                        saw_completed = true;
                        finished = true;
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::Provider;
    use serde_json::json;

    fn server() -> OpenAIResponsesServer {
        OpenAIResponsesServer::new(vec![UpstreamSpec::new(Provider::Anthropic, "k")])
    }

    #[test]
    fn stateless_method_flips_flag() {
        let s = server();
        assert!(!s.is_stateless());
        let s2 = s.stateless();
        assert!(s2.is_stateless());
    }

    #[tokio::test]
    async fn stateless_rejects_previous_response_id() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let s = server().stateless();
        tokio::spawn(async move {
            let _ = axum::serve(listener, s.router()).await;
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://{addr}/v1/responses"))
            .json(&json!({
                "model": "x",
                "input": "hi",
                "previous_response_id": "resp_anything",
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("stateless"),
            "got message: {}",
            body["error"]["message"]
        );
    }
}
