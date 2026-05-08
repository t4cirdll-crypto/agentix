//! Live smoke tests for Mimo (xiaomimimo) via the Anthropic-compatible
//! endpoint. Default-ignored — run with:
//!
//! ```bash
//! MIMO_API_KEY=tp-...  MIMO_BASE_URL=https://...  cargo test --test smoke_mimo -- --ignored
//! ```
//!
//! `MIMO_BASE_URL` is optional; defaults to the public api host
//! (`https://api.xiaomimimo.com/anthropic`) baked into the provider.
//!
//! Covers:
//!   1. Non-streaming `complete` — auth header, URL routing, response parsing.
//!   2. Streaming with thinking enabled — verifies `thinking: {type: enabled}`
//!      is accepted and reasoning text comes back.
//!   3. Multi-turn tool loop — verifies tool round-trip plus the
//!      `provider_data` envelope is wired up (signatures captured iff Mimo
//!      emits `signature_delta`, which the public spec doesn't promise).

use agentix::msg::LlmEvent;
use agentix::types::UsageStats;
use agentix::{
    Content, Message, ReasoningEffort, Request, Tool, ToolBundle, ToolCall, ToolOutput,
    UserContent, tool,
};
use futures::StreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};

fn key() -> Option<String> {
    std::env::var("MIMO_API_KEY").ok().filter(|k| !k.is_empty())
}

fn base() -> Option<String> {
    std::env::var("MIMO_BASE_URL")
        .ok()
        .filter(|s| !s.is_empty())
}

fn model() -> String {
    std::env::var("MIMO_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".into())
}

fn req(api_key: &str) -> Request {
    let mut r = Request::mimo(api_key).model(model());
    if let Some(b) = base() {
        r = r.base_url(b);
    }
    r
}

#[tokio::test]
#[ignore]
async fn mimo_complete_text() {
    let Some(k) = key() else {
        eprintln!("MIMO_API_KEY not set, skipping");
        return;
    };

    let http = reqwest::Client::new();
    let resp = req(&k)
        .reasoning_effort(ReasoningEffort::None)
        .user("In one short sentence: what is 2+2?")
        .complete(&http)
        .await
        .expect("complete should succeed");

    let content = resp.content.expect("must return content");
    assert!(!content.is_empty(), "content should be non-empty");
    eprintln!(
        "complete: {content:?} | usage prompt={} completion={} cache_read={}",
        resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.cache_read_tokens,
    );
}

#[tokio::test]
#[ignore]
async fn mimo_stream_thinking_enabled() {
    let Some(k) = key() else {
        eprintln!("MIMO_API_KEY not set, skipping");
        return;
    };

    let http = reqwest::Client::new();
    let mut stream = req(&k)
        .reasoning_effort(ReasoningEffort::High)
        .user("Briefly: how many vowels in 'mississippi'?")
        .stream(&http)
        .await
        .expect("stream should open");

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut usage: Option<UsageStats> = None;
    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => content.push_str(&t),
            LlmEvent::Reasoning(r) => reasoning.push_str(&r),
            LlmEvent::Usage(u) => usage = Some(u),
            LlmEvent::Error(e) => panic!("stream error: {e}"),
            LlmEvent::Done => break,
            _ => {}
        }
    }

    assert!(!content.is_empty(), "should return non-empty content");
    // Mimo's docs make extended thinking the default for v2.5/v2.5-pro/v2-pro/
    // v2-omni; with reasoning_effort(High) we expect reasoning text from those
    // models. v2-flash defaults to disabled — soft-assert so the test stays
    // useful across models.
    if reasoning.is_empty() {
        eprintln!("(no reasoning emitted — model may have thinking disabled)");
    } else {
        eprintln!("reasoning ({} chars): {reasoning}", reasoning.len());
    }
    if let Some(u) = usage {
        eprintln!(
            "usage: prompt={} completion={} cache_read={} reasoning={}",
            u.prompt_tokens, u.completion_tokens, u.cache_read_tokens, u.reasoning_tokens,
        );
    }
}

// ── Tool used by the multi-turn smoke test ───────────────────────────────────

struct LetterTool {
    counter: AtomicUsize,
}

#[tool]
impl agentix::Tool for LetterTool {
    /// Reveal the next character of the secret string stored inside this tool.
    /// Call repeatedly. Returns null once exhausted.
    async fn next_letter(&self) -> Option<String> {
        const WORD: [&str; 5] = ["m", "i", "m", "o", "!"];
        let idx = self.counter.fetch_add(1, Ordering::SeqCst);
        WORD.get(idx).map(|s| (*s).to_string())
    }
}

#[tokio::test]
#[ignore]
async fn mimo_multi_turn_tool_loop() {
    let Some(k) = key() else {
        eprintln!("MIMO_API_KEY not set, skipping");
        return;
    };

    let http = reqwest::Client::new();
    let bundle = ToolBundle::new().with(LetterTool {
        counter: AtomicUsize::new(0),
    });
    let raw_tools = bundle.raw_tools();

    let base = req(&k)
        .reasoning_effort(ReasoningEffort::High)
        .system_prompt(
            "You have one tool: `next_letter`. Each call reveals one character \
             of a secret string. Call it repeatedly until it returns null, then \
             state the assembled string in one short sentence.",
        );

    let mut history: Vec<Message> = vec![Message::User(vec![UserContent::Text {
        text: "What's stored in `next_letter`? Call it until null.".into(),
    }])];

    let mut signatures_round_tripped = false;
    let mut final_text = String::new();
    let mut total_usage = UsageStats::default();

    for _ in 0..8 {
        let mut stream = base
            .clone()
            .messages(history.clone())
            .tools(raw_tools.clone())
            .stream(&http)
            .await
            .expect("stream should open");

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut provider_state: Option<serde_json::Value> = None;

        while let Some(ev) = stream.next().await {
            match ev {
                LlmEvent::Token(t) => text.push_str(&t),
                LlmEvent::Reasoning(r) => reasoning.push_str(&r),
                LlmEvent::ToolCall(tc) => tool_calls.push(tc),
                LlmEvent::AssistantState(v) => {
                    if let Some(blocks) = v.get("anthropic_content").and_then(|b| b.as_array())
                        && blocks.iter().any(|b| b.get("signature").is_some())
                    {
                        signatures_round_tripped = true;
                    }
                    provider_state = Some(v);
                }
                LlmEvent::Usage(u) => total_usage += u,
                LlmEvent::Error(e) => panic!("stream error: {e}"),
                LlmEvent::Done => break,
                _ => {}
            }
        }

        history.push(Message::Assistant {
            content: if text.is_empty() {
                None
            } else {
                Some(text.clone())
            },
            reasoning: if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            },
            tool_calls: tool_calls.clone(),
            provider_data: provider_state,
        });

        if tool_calls.is_empty() {
            final_text = text;
            break;
        }

        for tc in tool_calls {
            let args: serde_json::Value =
                serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
            let mut out = bundle.call(&tc.name, args).await;
            let mut content: Vec<Content> = vec![];
            while let Some(o) = out.next().await {
                if let ToolOutput::Result(v) = o {
                    content = v;
                }
            }
            history.push(Message::ToolResult {
                call_id: tc.id,
                content,
            });
        }
    }

    assert!(
        !final_text.is_empty(),
        "model should produce a final text turn after exhausting the tool"
    );
    assert!(
        final_text.contains("mimo!") || final_text.contains("mimo"),
        "final answer should reveal the assembled string 'mimo!', got: {final_text}"
    );
    eprintln!("final answer: {final_text}");
    eprintln!(
        "tool-loop usage: prompt={} completion={} cache_read={} reasoning={}",
        total_usage.prompt_tokens,
        total_usage.completion_tokens,
        total_usage.cache_read_tokens,
        total_usage.reasoning_tokens,
    );
    eprintln!("thinking signatures round-tripped: {signatures_round_tripped}");
}
