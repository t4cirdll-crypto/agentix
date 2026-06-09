//! Integration tests using an embedded Axum mock LLM server with real response
//! fixtures. Covers all 4 providers (OpenAI, DeepSeek, Anthropic, Gemini) for
//! both streaming (SSE) and non-streaming (JSON) completions, plus edge cases
//! like slow streams, malformed SSE, HTTP errors, and retry behaviour.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agentix::msg::LlmEvent;
use agentix::{Provider, Request};
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Uri};
use axum::response::Response;
use futures::StreamExt;
use tokio::net::TcpListener;

// ═══════════════════════════════════════════════════════════════════════════════
//  MOCK SERVER
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
enum MockBehaviour {
    Sse(String),
    Json(String),
    SlowSse { body: String, chunk_delay: Duration },
    Error { status: u16, body: String },
}

#[derive(Clone)]
struct MockState {
    behaviour: MockBehaviour,
}

async fn handle(State(state): State<MockState>) -> Response {
    match state.behaviour {
        MockBehaviour::Sse(body) => Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(Body::from(body))
            .unwrap(),
        MockBehaviour::Json(body) => Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap(),
        MockBehaviour::SlowSse { body, chunk_delay } => {
            let stream = async_stream::stream! {
                for line in body.lines() {
                    tokio::time::sleep(chunk_delay).await;
                    yield Ok::<_, std::convert::Infallible>(format!("{line}\n"));
                }
            };
            Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .header("cache-control", "no-cache")
                .body(Body::from_stream(stream))
                .unwrap()
        }
        MockBehaviour::Error { status, body } => Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap(),
    }
}

async fn start_mock(behaviour: MockBehaviour) -> String {
    let state = MockState { behaviour };
    let app = Router::new().fallback(handle).with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[derive(Clone)]
struct CaptureState {
    body: Arc<Mutex<Option<serde_json::Value>>>,
    response: String,
}

async fn handle_capture(State(state): State<CaptureState>, body: String) -> Response {
    let parsed: serde_json::Value =
        serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body));
    *state.body.lock().unwrap() = Some(parsed);
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(state.response))
        .unwrap()
}

async fn start_capture_mock(response: String) -> (String, Arc<Mutex<Option<serde_json::Value>>>) {
    let body = Arc::new(Mutex::new(None));
    let state = CaptureState {
        body: body.clone(),
        response,
    };
    let app = Router::new().fallback(handle_capture).with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), body)
}

#[derive(Clone)]
struct CaptureRequest {
    path: String,
    headers: Vec<(String, String)>,
    body: serde_json::Value,
}

#[derive(Clone)]
struct CaptureRequestState {
    request: Arc<Mutex<Option<CaptureRequest>>>,
    response: String,
}

async fn handle_request_capture(
    State(state): State<CaptureRequestState>,
    uri: Uri,
    headers: HeaderMap,
    body: String,
) -> Response {
    let parsed: serde_json::Value =
        serde_json::from_str(&body).unwrap_or(serde_json::Value::String(body));
    let headers = headers
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                value.to_str().unwrap_or("<non-utf8>").to_string(),
            )
        })
        .collect();
    *state.request.lock().unwrap() = Some(CaptureRequest {
        path: uri.path().to_string(),
        headers,
        body: parsed,
    });
    Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(Body::from(state.response))
        .unwrap()
}

async fn start_request_capture_mock(
    response: String,
) -> (String, Arc<Mutex<Option<CaptureRequest>>>) {
    let request = Arc::new(Mutex::new(None));
    let state = CaptureRequestState {
        request: request.clone(),
        response,
    };
    let app = Router::new()
        .fallback(handle_request_capture)
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), request)
}

fn fixture(path: &str) -> String {
    let full = format!("{}/tests/fixtures/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("fixture {full}: {e}"))
}

fn collect_tokens(events: &[LlmEvent]) -> String {
    events
        .iter()
        .filter_map(|e| {
            if let LlmEvent::Token(t) = e {
                Some(t.as_str())
            } else {
                None
            }
        })
        .collect()
}

fn collect_reasoning(events: &[LlmEvent]) -> String {
    events
        .iter()
        .filter_map(|e| {
            if let LlmEvent::Reasoning(r) = e {
                Some(r.as_str())
            } else {
                None
            }
        })
        .collect()
}

fn find_usage(events: &[LlmEvent]) -> Option<&agentix::types::UsageStats> {
    events.iter().find_map(|e| {
        if let LlmEvent::Usage(u) = e {
            Some(u)
        } else {
            None
        }
    })
}

fn http() -> reqwest::Client {
    // no_proxy ensures 127.0.0.1 isn't routed through system proxy (e.g. Clash)
    reqwest::Client::builder().no_proxy().build().unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════════
//  OPENAI
// ═══════════════════════════════════════════════════════════════════════════════

mod openai {
    use super::*;

    fn req(base_url: &str) -> Request {
        Request::new(Provider::OpenAI, "test-key")
            .base_url(base_url)
            .model("gpt-4o-test")
            .user("hi")
    }

    #[tokio::test]
    async fn stream_text() {
        let url = start_mock(MockBehaviour::Sse(fixture("openai/stream_text.sse"))).await;
        let events: Vec<_> = req(&url).stream(&http()).await.unwrap().collect().await;

        assert_eq!(collect_tokens(&events), "The capital of France is Paris.");
        let u = find_usage(&events).expect("should have usage");
        assert_eq!(u.prompt_tokens, 14);
        assert_eq!(u.completion_tokens, 7);
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn stream_tool_call() {
        let url = start_mock(MockBehaviour::Sse(fixture("openai/stream_tool_call.sse"))).await;
        let events: Vec<_> = req(&url)
            .user("weather?")
            .stream(&http())
            .await
            .unwrap()
            .collect()
            .await;

        let tool_calls: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let LlmEvent::ToolCall(tc) = e {
                    Some(tc)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "call_abc123");
        assert_eq!(
            tool_calls[0].arguments,
            r#"{"city":"Tokyo","units":"celsius"}"#
        );
    }

    #[tokio::test]
    async fn complete_text() {
        let url = start_mock(MockBehaviour::Json(fixture("openai/complete_text.json"))).await;
        let resp = req(&url).user("q").complete(&http()).await.unwrap();

        assert_eq!(
            resp.content.as_deref(),
            Some("The capital of France is Paris.")
        );
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.usage.total_tokens, 21);
    }

    #[tokio::test]
    async fn complete_tool_call() {
        let url = start_mock(MockBehaviour::Json(fixture(
            "openai/complete_tool_call.json",
        )))
        .await;
        let resp = req(&url).user("weather?").complete(&http()).await.unwrap();

        assert!(resp.content.is_none());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].id, "call_abc123");
        // Reasoning tokens from output_tokens_details.
        assert_eq!(resp.usage.reasoning_tokens, 3);
        // provider_data captures the full output[] (reasoning + function_call)
        // verbatim — includes encrypted_content for next-turn round-trip.
        let pd = resp
            .provider_data
            .as_ref()
            .expect("reasoning+tool_call turn must emit provider_data");
        let items = pd["openai_responses_items"]
            .as_array()
            .expect("envelope must carry an array");
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[0]["encrypted_content"], "ENC_OPAQUE_1");
        assert_eq!(items[1]["type"], "function_call");
    }

    #[tokio::test]
    async fn stream_reasoning_without_tool_does_not_emit_state() {
        let url = start_mock(MockBehaviour::Sse(fixture("openai/stream_reasoning.sse"))).await;
        let events: Vec<_> = req(&url).stream(&http()).await.unwrap().collect().await;

        assert_eq!(collect_reasoning(&events), "Let me think about this...");
        assert_eq!(collect_tokens(&events), "The answer is 42.");
        // Gate closed: pure reasoning→message turn, no function_call → no
        // round-trip needed.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, LlmEvent::AssistantState(_))),
            "pure reasoning turn must not emit AssistantState"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DEEPSEEK (shares OpenAI format, but exercises DeepSeek provider path)
// ═══════════════════════════════════════════════════════════════════════════════

mod deepseek {
    use super::*;

    fn req(base_url: &str) -> Request {
        Request::new(Provider::DeepSeek, "test-key")
            .base_url(base_url)
            .model("deepseek-test")
            .user("hi")
    }

    #[tokio::test]
    async fn stream_text() {
        let url = start_mock(MockBehaviour::Sse(fixture(
            "chat_completions/stream_text.sse",
        )))
        .await;
        let events: Vec<_> = req(&url).stream(&http()).await.unwrap().collect().await;

        assert_eq!(collect_tokens(&events), "The capital of France is Paris.");
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn stream_reasoning() {
        let url = start_mock(MockBehaviour::Sse(fixture(
            "chat_completions/stream_reasoning.sse",
        )))
        .await;
        let events: Vec<_> = req(&url)
            .user("q")
            .stream(&http())
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(collect_reasoning(&events), "Let me think about this...");
        assert_eq!(collect_tokens(&events), "The answer is 42.");
    }

    #[tokio::test]
    async fn complete_text() {
        let url = start_mock(MockBehaviour::Json(fixture(
            "chat_completions/complete_text.json",
        )))
        .await;
        let resp = req(&url).user("q").complete(&http()).await.unwrap();

        assert_eq!(
            resp.content.as_deref(),
            Some("The capital of France is Paris.")
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  ANTHROPIC
// ═══════════════════════════════════════════════════════════════════════════════

mod anthropic {
    use super::*;

    fn req(base_url: &str) -> Request {
        Request::new(Provider::Anthropic, "test-key")
            .base_url(base_url)
            .model("claude-test")
            .user("hi")
    }

    #[tokio::test]
    async fn stream_text() {
        let url = start_mock(MockBehaviour::Sse(fixture("anthropic/stream_text.sse"))).await;
        let events: Vec<_> = req(&url).stream(&http()).await.unwrap().collect().await;

        assert_eq!(collect_tokens(&events), "The capital of France is Paris.");
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn stream_tool_call() {
        let url = start_mock(MockBehaviour::Sse(fixture(
            "anthropic/stream_tool_call.sse",
        )))
        .await;
        let events: Vec<_> = req(&url)
            .user("weather?")
            .stream(&http())
            .await
            .unwrap()
            .collect()
            .await;

        let tool_calls: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let LlmEvent::ToolCall(tc) = e {
                    Some(tc)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
        assert_eq!(tool_calls[0].id, "toolu_abc123");
        assert_eq!(
            tool_calls[0].arguments,
            r#"{"city":"Tokyo","units":"celsius"}"#
        );
    }

    #[tokio::test]
    async fn complete_text() {
        let url = start_mock(MockBehaviour::Json(fixture("anthropic/complete_text.json"))).await;
        let resp = req(&url).user("q").complete(&http()).await.unwrap();

        assert_eq!(
            resp.content.as_deref(),
            Some("The capital of France is Paris.")
        );
        assert_eq!(resp.usage.prompt_tokens, 12);
        assert_eq!(resp.usage.completion_tokens, 8);
    }

    #[tokio::test]
    async fn complete_tool_call() {
        let url = start_mock(MockBehaviour::Json(fixture(
            "anthropic/complete_tool_call.json",
        )))
        .await;
        let resp = req(&url).user("weather?").complete(&http()).await.unwrap();

        assert!(resp.content.is_none());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].id, "toolu_abc123");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MIMO (Anthropic-compatible Messages API)
// ═══════════════════════════════════════════════════════════════════════════════

mod mimo {
    use super::*;

    fn req(base_url: &str) -> Request {
        Request::new(Provider::Mimo, "test-key")
            .base_url(base_url)
            .model("mimo-v2.5-pro")
            .user("hi")
    }

    #[tokio::test]
    async fn complete_text_uses_anthropic_wire_format_with_mimo_auth() {
        let (url, captured) =
            start_request_capture_mock(fixture("anthropic/complete_text.json")).await;
        let resp = req(&url).complete(&http()).await.unwrap();

        assert_eq!(
            resp.content.as_deref(),
            Some("The capital of France is Paris.")
        );

        let captured = captured
            .lock()
            .unwrap()
            .clone()
            .expect("mock server should capture request");
        assert_eq!(captured.path, "/v1/messages");
        assert_eq!(captured.body["model"], "mimo-v2.5-pro");
        assert_eq!(captured.body["messages"][0]["role"], "user");
        assert!(
            captured
                .headers
                .iter()
                .any(|(name, value)| name == "api-key" && value == "test-key")
        );
        assert!(!captured.headers.iter().any(|(name, _)| name == "x-api-key"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  GEMINI
// ═══════════════════════════════════════════════════════════════════════════════

mod gemini {
    use super::*;

    fn req(base_url: &str) -> Request {
        Request::new(Provider::Gemini, "test-key")
            .base_url(base_url)
            .model("gemini-test")
            .user("hi")
    }

    #[tokio::test]
    async fn stream_text() {
        let url = start_mock(MockBehaviour::Sse(fixture("gemini/stream_text.sse"))).await;
        let events: Vec<_> = req(&url).stream(&http()).await.unwrap().collect().await;

        assert_eq!(collect_tokens(&events), "The capital of France is Paris.");
        let u = find_usage(&events).expect("should have usage");
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 7);
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn stream_tool_call() {
        let url = start_mock(MockBehaviour::Sse(fixture("gemini/stream_tool_call.sse"))).await;
        let events: Vec<_> = req(&url)
            .user("weather?")
            .stream(&http())
            .await
            .unwrap()
            .collect()
            .await;

        let tool_calls: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let LlmEvent::ToolCall(tc) = e {
                    Some(tc)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "get_weather");
    }

    #[tokio::test]
    async fn complete_text() {
        let url = start_mock(MockBehaviour::Json(fixture("gemini/complete_text.json"))).await;
        let resp = req(&url).user("q").complete(&http()).await.unwrap();

        assert_eq!(
            resp.content.as_deref(),
            Some("The capital of France is Paris.")
        );
        assert_eq!(resp.usage.total_tokens, 17);
    }

    #[tokio::test]
    async fn complete_thinking_tool_call_captures_signatures() {
        let url = start_mock(MockBehaviour::Json(fixture(
            "gemini/complete_thinking_tool_call.json",
        )))
        .await;
        let resp = Request::new(Provider::Gemini, "test-key")
            .base_url(&url)
            .model("gemini-3-pro")
            .user("weather?")
            .complete(&http())
            .await
            .unwrap();
        assert_eq!(
            resp.reasoning.as_deref(),
            Some("I need to check the weather for Paris.")
        );
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.usage.reasoning_tokens, 4);
        let pd = resp
            .provider_data
            .as_ref()
            .expect("thinking+tool_call must emit provider_data");
        let parts = pd["gemini_parts"].as_array().expect("array");
        assert_eq!(parts[0]["thoughtSignature"], "THOUGHT_SIG_A");
        assert_eq!(parts[1]["thoughtSignature"], "FUNC_CALL_SIG_A");
        assert_eq!(parts[1]["functionCall"]["name"], "get_weather");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

mod edge_cases {
    use super::*;

    #[tokio::test]
    async fn slow_sse_still_delivers_all_tokens() {
        // Chat Completions edge case — exercised on OpenRouter (which keeps
        // that wire format). `Provider::OpenAI` now speaks Responses API.
        let url = start_mock(MockBehaviour::SlowSse {
            body: fixture("chat_completions/stream_text.sse"),
            chunk_delay: Duration::from_millis(20),
        })
        .await;

        let events: Vec<_> = Request::new(Provider::OpenRouter, "test-key")
            .base_url(&url)
            .model("openai/gpt-4o")
            .user("hi")
            .stream(&http())
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(collect_tokens(&events), "The capital of France is Paris.");
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn http_401_error_propagates() {
        let url = start_mock(MockBehaviour::Error {
            status: 401,
            body: r#"{"error":"invalid_api_key"}"#.into(),
        })
        .await;

        let result = Request::new(Provider::OpenAI, "bad-key")
            .base_url(&url)
            .user("hi")
            .stream(&http())
            .await;
        let err = result.err().expect("should be an error");
        let msg = err.to_string();
        assert!(
            msg.contains("401") || msg.contains("invalid_api_key"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn http_429_retries_and_fails() {
        let url = start_mock(MockBehaviour::Error {
            status: 429,
            body: r#"{"error":"rate_limited"}"#.into(),
        })
        .await;

        let err = Request::new(Provider::OpenAI, "key")
            .base_url(&url)
            .model("gpt-test")
            .retries(2, 10)
            .user("hi")
            .complete(&http())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("429"), "got: {}", err);
    }

    #[tokio::test]
    async fn http_500_error_on_complete() {
        let url = start_mock(MockBehaviour::Error {
            status: 500,
            body: r#"{"error":"internal_server_error"}"#.into(),
        })
        .await;

        let err = Request::new(Provider::OpenAI, "key")
            .base_url(&url)
            .model("gpt-test")
            .retries(1, 10)
            .user("hi")
            .complete(&http())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("500"), "got: {}", err);
    }

    #[tokio::test]
    async fn empty_sse_stream_yields_done() {
        let url = start_mock(MockBehaviour::Sse("data: [DONE]\n\n".into())).await;

        let events: Vec<_> = Request::new(Provider::OpenAI, "key")
            .base_url(&url)
            .user("hi")
            .stream(&http())
            .await
            .unwrap()
            .collect()
            .await;

        assert!(collect_tokens(&events).is_empty());
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn malformed_sse_chunk_is_skipped() {
        // Chat Completions edge case — use OpenRouter which keeps that wire.
        let body = "data: {\"bad json\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";
        let url = start_mock(MockBehaviour::Sse(body.into())).await;

        let events: Vec<_> = Request::new(Provider::OpenRouter, "key")
            .base_url(&url)
            .user("hi")
            .stream(&http())
            .await
            .unwrap()
            .collect()
            .await;

        assert_eq!(collect_tokens(&events), "ok");
        assert!(matches!(events.last(), Some(LlmEvent::Done)));
    }

    #[tokio::test]
    async fn complete_empty_choices() {
        let body =
            r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}"#;
        let url = start_mock(MockBehaviour::Json(body.into())).await;

        let resp = Request::new(Provider::OpenRouter, "key")
            .base_url(&url)
            .user("hi")
            .complete(&http())
            .await
            .unwrap();
        assert!(resp.content.is_none());
        assert!(resp.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn anthropic_complete_with_thinking() {
        let body = r#"{
            "id": "msg_think1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "42"}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }"#;
        let url = start_mock(MockBehaviour::Json(body.into())).await;

        let resp = Request::new(Provider::Anthropic, "key")
            .base_url(&url)
            .user("meaning of life")
            .complete(&http())
            .await
            .unwrap();
        assert_eq!(resp.content.as_deref(), Some("42"));
        assert_eq!(
            resp.reasoning.as_deref(),
            Some("Let me reason about this...")
        );
    }
}
