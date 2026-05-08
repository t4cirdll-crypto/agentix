//! Integration tests for the Anthropic Messages-compatible server.
//!
//! Each test stands up an in-process axum mock that fakes an Anthropic
//! upstream, then points an `AnthropicServer` at it and drives requests
//! through the public HTTP surface.

#![cfg(feature = "server-anthropic")]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use agentix::Provider;
use agentix::server::{AnthropicServer, UpstreamSpec};
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use futures::StreamExt;
use futures::stream::{self};
use serde_json::{Value, json};
use tokio::net::TcpListener;

// ── Mock upstream helpers ────────────────────────────────────────────────────

/// Configurable mock that pretends to be an Anthropic Messages endpoint.
#[derive(Clone, Default)]
struct MockUpstream {
    /// Number of requests received.
    hits: Arc<AtomicUsize>,
    /// If true, return 503 instead of streaming.
    fail: Arc<std::sync::atomic::AtomicBool>,
    /// SSE events to emit when the request is `stream:true`.
    sse_events: Arc<Vec<MockEvent>>,
    /// JSON body for non-streaming responses.
    json_body: Arc<Value>,
}

#[derive(Debug, Clone)]
enum MockEvent {
    MessageStart,
    TextBlockStart,
    TextDelta(String),
    BlockStop,
    MessageDelta { stop: &'static str },
    MessageStop,
}

impl MockEvent {
    fn to_pair(&self, index: u32) -> (&'static str, Value) {
        match self {
            MockEvent::MessageStart => (
                "message_start",
                json!({
                    "type":"message_start",
                    "message":{
                        "id":"msg_mock_1",
                        "type":"message",
                        "role":"assistant",
                        "model":"mock",
                        "content":[],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {"input_tokens": 1, "output_tokens": 0}
                    }
                }),
            ),
            MockEvent::TextBlockStart => (
                "content_block_start",
                json!({
                    "type":"content_block_start",
                    "index": index,
                    "content_block":{"type":"text","text":""}
                }),
            ),
            MockEvent::TextDelta(t) => (
                "content_block_delta",
                json!({
                    "type":"content_block_delta",
                    "index": index,
                    "delta":{"type":"text_delta","text": t}
                }),
            ),
            MockEvent::BlockStop => (
                "content_block_stop",
                json!({"type":"content_block_stop","index": index}),
            ),
            MockEvent::MessageDelta { stop } => (
                "message_delta",
                json!({
                    "type":"message_delta",
                    "delta":{"stop_reason": stop, "stop_sequence": Value::Null},
                    "usage":{"output_tokens": 5}
                }),
            ),
            MockEvent::MessageStop => ("message_stop", json!({"type":"message_stop"})),
        }
    }
}

async fn mock_messages_handler(
    State(mock): State<MockUpstream>,
    body: axum::Json<Value>,
) -> Response {
    mock.hits.fetch_add(1, Ordering::SeqCst);
    if mock.fail.load(Ordering::SeqCst) {
        return (StatusCode::SERVICE_UNAVAILABLE, "boom").into_response();
    }
    let stream_requested = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !stream_requested {
        return (StatusCode::OK, axum::Json((*mock.json_body).clone())).into_response();
    }
    let events: Vec<MockEvent> = mock.sse_events.iter().cloned().collect();
    let s = stream::iter(events.into_iter().map(|ev| {
        let (name, payload) = ev.to_pair(0);
        Ok::<_, std::convert::Infallible>(Event::default().event(name).data(payload.to_string()))
    }));
    Sse::new(s.boxed())
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(60)))
        .into_response()
}

async fn spawn_mock_upstream(mock: MockUpstream) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let app: Router = Router::new()
        .route("/v1/messages", post(mock_messages_handler))
        .with_state(mock);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (addr, handle)
}

async fn spawn_proxy(chain: Vec<UpstreamSpec>) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let server = AnthropicServer::new(chain);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, server.router()).await;
    });
    (addr, handle)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn non_streaming_text_round_trip() {
    let mock = MockUpstream {
        json_body: Arc::new(json!({
            "id":"msg_x","type":"message","role":"assistant","model":"mock",
            "content":[{"type":"text","text":"hello world"}],
            "usage":{"input_tokens":3,"output_tokens":2},
            "stop_reason":"end_turn"
        })),
        ..Default::default()
    };
    let (mock_addr, _h1) = spawn_mock_upstream(mock.clone()).await;

    let chain = vec![
        UpstreamSpec::new(Provider::Anthropic, "k").with_base_url(format!("http://{mock_addr}")),
    ];
    let (proxy_addr, _h2) = spawn_proxy(chain).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/messages"))
        .json(&json!({
            "model":"claude-haiku-4-5",
            "max_tokens":256,
            "messages":[{"role":"user","content":"hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    let text = body["content"][0]["text"].as_str().unwrap();
    assert_eq!(text, "hello world");
    assert_eq!(body["stop_reason"], "end_turn");
}

#[tokio::test]
async fn streaming_text_round_trip() {
    let mock = MockUpstream {
        sse_events: Arc::new(vec![
            MockEvent::MessageStart,
            MockEvent::TextBlockStart,
            MockEvent::TextDelta("hi".into()),
            MockEvent::TextDelta(" there".into()),
            MockEvent::BlockStop,
            MockEvent::MessageDelta { stop: "end_turn" },
            MockEvent::MessageStop,
        ]),
        ..Default::default()
    };
    let (mock_addr, _h1) = spawn_mock_upstream(mock.clone()).await;

    let chain = vec![
        UpstreamSpec::new(Provider::Anthropic, "k").with_base_url(format!("http://{mock_addr}")),
    ];
    let (proxy_addr, _h2) = spawn_proxy(chain).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/messages"))
        .json(&json!({
            "model":"claude-haiku-4-5",
            "max_tokens":256,
            "stream":true,
            "messages":[{"role":"user","content":"hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let bytes = resp.bytes().await.unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();

    // Sanity-check the SSE shape: must contain message_start, text deltas, message_stop.
    assert!(text.contains("event: message_start"), "missing message_start: {text}");
    assert!(text.contains("event: content_block_start"), "missing content_block_start: {text}");
    assert!(text.contains("event: content_block_delta"));
    assert!(text.contains("\"text\":\"hi\""));
    assert!(text.contains("event: content_block_stop"));
    assert!(text.contains("event: message_delta"));
    assert!(text.contains("event: message_stop"));
}

#[tokio::test]
async fn fallback_to_secondary_when_primary_5xx_non_streaming() {
    let primary_mock = MockUpstream {
        fail: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        ..Default::default()
    };
    let secondary_mock = MockUpstream {
        json_body: Arc::new(json!({
            "id":"msg_y","type":"message","role":"assistant","model":"mock",
            "content":[{"type":"text","text":"fallback"}],
            "usage":{"input_tokens":1,"output_tokens":1},
            "stop_reason":"end_turn"
        })),
        ..Default::default()
    };
    let (primary_addr, _h1) = spawn_mock_upstream(primary_mock.clone()).await;
    let (secondary_addr, _h2) = spawn_mock_upstream(secondary_mock.clone()).await;

    let chain = vec![
        UpstreamSpec::new(Provider::Anthropic, "k1").with_base_url(format!("http://{primary_addr}")),
        UpstreamSpec::new(Provider::Anthropic, "k2").with_base_url(format!("http://{secondary_addr}")),
    ];
    let (proxy_addr, _h3) = spawn_proxy(chain).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/messages"))
        .json(&json!({
            "model":"x","max_tokens":256,
            "messages":[{"role":"user","content":"hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["content"][0]["text"], "fallback");
    assert!(primary_mock.hits.load(Ordering::SeqCst) >= 1);
    assert!(secondary_mock.hits.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn count_tokens_returns_estimate() {
    let chain = vec![UpstreamSpec::new(Provider::Anthropic, "k")];
    let (proxy_addr, _h) = spawn_proxy(chain).await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{proxy_addr}/v1/messages/count_tokens"))
        .json(&json!({
            "model":"x","max_tokens":100,
            "messages":[{"role":"user","content":"hello world"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["input_tokens"].as_u64().unwrap_or(0) > 0);
}

#[tokio::test]
async fn unknown_path_returns_anthropic_shape_404() {
    let chain = vec![UpstreamSpec::new(Provider::Anthropic, "k")];
    let (proxy_addr, _h) = spawn_proxy(chain).await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{proxy_addr}/v1/messages/batches"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "not_found_error");
}
