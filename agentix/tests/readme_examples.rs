//! Compile-time verification of every code example in README.md.
//! Each test mirrors a README snippet to ensure the documented API is real.

use agentix::request::ToolCall;
use agentix::{LlmEvent, Message, Provider, ReasoningEffort, Request, UserContent};

// ── Quick Start (simplified, no network) ──────────────────────────────────────

#[test]
fn quickstart_compiles() {
    let _http = reqwest::Client::new();

    let _req = Request::new(Provider::DeepSeek, "sk-test")
        .system_prompt("You are a helpful assistant.")
        .user("What is the capital of France?");
    // .stream(&http) would need network — just verify the builder compiles
}

// ── Providers section ─────────────────────────────────────────────────────────

#[test]
fn provider_constructors() {
    let req = Request::new(Provider::DeepSeek, "sk-...");
    assert_eq!(req.model, "deepseek-chat");

    let req = Request::new(Provider::OpenAI, "sk-...");
    assert_eq!(req.model, "gpt-4o");

    let req = Request::new(Provider::Anthropic, "sk-ant-...");
    assert_eq!(req.model, "claude-sonnet-4-20250514");

    let req = Request::new(Provider::Gemini, "AIza...");
    assert_eq!(req.model, "gemini-2.0-flash");

    let req = Request::new(Provider::Kimi, "sk-...");
    assert_eq!(req.model, "kimi-k2.5");

    let req = Request::new(Provider::Glm, "sk-...");
    assert_eq!(req.model, "glm-5");

    let req = Request::new(Provider::Minimax, "sk-...");
    assert_eq!(req.model, "MiniMax-M2.7");

    let req = Request::new(Provider::Mimo, "sk-...");
    assert_eq!(req.model, "mimo-v2.5-pro");

    let req = Request::new(Provider::Grok, "sk-...");
    assert_eq!(req.model, "grok-4");

    let req = Request::new(Provider::OpenRouter, "sk-or-...");
    assert_eq!(req.model, "openrouter/auto");

    let req = Request::openrouter("local-key")
        .base_url("http://localhost:11434/v1")
        .model("llama3.1");
    assert_eq!(req.base_url, "http://localhost:11434/v1");
    assert_eq!(req.model, "llama3.1");
}

// ── Builder methods section ───────────────────────────────────────────────────

#[test]
fn builder_methods() {
    let msg = Message::Assistant {
        content: Some("hi".into()),
        reasoning: None,
        tool_calls: vec![],
        provider_data: None,
    };

    let req = Request::new(Provider::DeepSeek, "sk-...")
        .model("deepseek-reasoner")
        .base_url("https://custom.api/v1")
        .system_prompt("You are helpful.")
        .max_tokens(4096)
        .temperature(0.7)
        .reasoning_effort(ReasoningEffort::High)
        .retries(5, 2000)
        .user("Hello!")
        .message(msg)
        .messages(vec![Message::User(vec![UserContent::Text {
            text: "test".into(),
        }])])
        .tools(vec![]);

    assert_eq!(req.model, "deepseek-reasoner");
    assert_eq!(req.base_url, "https://custom.api/v1");
    assert_eq!(req.system_message.as_deref(), Some("You are helpful."));
    assert_eq!(req.max_tokens, Some(4096));
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.reasoning_effort, Some(ReasoningEffort::High));
    assert_eq!(req.max_retries, 5);
    assert_eq!(req.retry_delay_ms, 2000);
}

// ── LlmEvent variants exist ──────────────────────────────────────────────────

#[test]
fn llm_event_variants_exist() {
    // Verify every variant mentioned in README compiles
    let _ = LlmEvent::Token("hello".into());
    let _ = LlmEvent::Reasoning("think".into());
    let _ = LlmEvent::ToolCall(ToolCall {
        id: "c1".into(),
        name: "f".into(),
        arguments: "{}".into(),
    });
    let _ = LlmEvent::Usage(agentix::UsageStats::default());
    let _ = LlmEvent::Done;
    let _ = LlmEvent::Error("err".into());
    let _ = LlmEvent::ToolCallChunk(agentix::types::ToolCallChunk {
        id: "c1".into(),
        name: "f".into(),
        delta: "{}".into(),
        index: 0,
    });
}

// ── CompleteResponse fields ───────────────────────────────────────────────────

#[test]
fn complete_response_fields() {
    let resp = agentix::CompleteResponse::default();
    let _ = resp.content;
    let _ = resp.reasoning;
    let _ = resp.tool_calls;
    let _ = resp.usage;
    let _ = resp.finish_reason;
}
