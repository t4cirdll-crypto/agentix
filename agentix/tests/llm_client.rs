//! Tests for the `Request` builder API.
//!
//! Covers constructors, builder setters, provider defaults, and field access.

use agentix::request::{Message, UserContent};
use agentix::{Provider, Request};
use serde_json::json;

fn user_msg(text: &str) -> Message {
    Message::User(vec![UserContent::Text { text: text.into() }])
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PROVIDER DEFAULTS
// ═══════════════════════════════════════════════════════════════════════════════

mod provider_defaults {
    use super::*;

    #[test]
    fn deepseek_defaults() {
        let r = Request::new(Provider::DeepSeek, "sk-test");
        assert_eq!(r.provider, Provider::DeepSeek);
        assert_eq!(r.model, "deepseek-chat");
        assert_eq!(r.base_url, "https://api.deepseek.com");
        assert!(r.system_message.is_none());
        assert!(r.max_tokens.is_none());
        assert!(r.temperature.is_none());
    }

    #[test]
    fn openai_defaults() {
        let r = Request::new(Provider::OpenAI, "sk-test");
        assert_eq!(r.model, "gpt-4o");
    }

    #[test]
    fn anthropic_defaults() {
        let r = Request::new(Provider::Anthropic, "sk-test");
        assert_eq!(r.model, "claude-sonnet-4-20250514");
    }

    #[test]
    fn gemini_defaults() {
        let r = Request::new(Provider::Gemini, "sk-test");
        assert_eq!(r.model, "gemini-2.0-flash");
    }

    #[test]
    fn mimo_defaults() {
        let r = Request::new(Provider::Mimo, "sk-test");
        assert_eq!(r.provider, Provider::Mimo);
        assert_eq!(r.model, "mimo-v2.5-pro");
        assert_eq!(r.base_url, "https://api.xiaomimimo.com/anthropic");
    }

    #[test]
    fn effective_base_url_uses_default_when_none() {
        let r = Request::new(Provider::DeepSeek, "k");
        assert_eq!(r.effective_base_url(), "https://api.deepseek.com");
    }

    #[test]
    fn effective_base_url_uses_override() {
        let r = Request::new(Provider::DeepSeek, "k").base_url("https://custom.api");
        assert_eq!(r.effective_base_url(), "https://custom.api");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  BUILDER SETTERS
// ═══════════════════════════════════════════════════════════════════════════════

mod builder {
    use super::*;

    #[test]
    fn model_setter() {
        let r = Request::new(Provider::DeepSeek, "k").model("custom-model");
        assert_eq!(r.model, "custom-model");
    }

    #[test]
    fn system_prompt_setter() {
        let r = Request::new(Provider::DeepSeek, "k").system_prompt("You are helpful.");
        assert_eq!(r.system_message.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn max_tokens_setter() {
        let r = Request::new(Provider::DeepSeek, "k").max_tokens(2048);
        assert_eq!(r.max_tokens, Some(2048));
    }

    #[test]
    fn temperature_setter() {
        let r = Request::new(Provider::DeepSeek, "k").temperature(0.7);
        assert_eq!(r.temperature, Some(0.7));
    }

    #[test]
    fn base_url_setter() {
        let r = Request::new(Provider::OpenAI, "k").base_url("https://new.url");
        assert_eq!(r.base_url, "https://new.url");
    }

    #[test]
    fn chained_setters() {
        let r = Request::new(Provider::DeepSeek, "k")
            .model("m1")
            .base_url("https://chain.test")
            .system_prompt("sp")
            .max_tokens(100)
            .temperature(0.5);
        assert_eq!(r.model, "m1");
        assert_eq!(r.base_url, "https://chain.test");
        assert_eq!(r.system_message.as_deref(), Some("sp"));
        assert_eq!(r.max_tokens, Some(100));
        assert_eq!(r.temperature, Some(0.5));
    }

    #[test]
    fn user_convenience() {
        let r = Request::new(Provider::DeepSeek, "k").user("hello");
        assert_eq!(r.messages.len(), 1);
        assert!(matches!(&r.messages[0], Message::User(parts) if parts.len() == 1));
    }

    #[test]
    fn push_message() {
        let r = Request::new(Provider::DeepSeek, "k")
            .message(user_msg("a"))
            .message(user_msg("b"));
        assert_eq!(r.messages.len(), 2);
    }

    #[test]
    fn messages_setter() {
        let msgs = vec![user_msg("a"), user_msg("b"), user_msg("c")];
        let r = Request::new(Provider::DeepSeek, "k").messages(msgs);
        assert_eq!(r.messages.len(), 3);
    }

    #[test]
    fn extra_body_setter() {
        let mut extra = serde_json::Map::new();
        extra.insert("foo".into(), json!("bar"));
        let r = Request::new(Provider::DeepSeek, "k").extra_body(extra.clone());
        assert_eq!(r.extra_body.get("foo").unwrap(), "bar");
    }

    #[test]
    fn retry_settings() {
        let r = Request::new(Provider::DeepSeek, "k");
        // Check defaults
        assert_eq!(r.max_retries, 3);
        assert_eq!(r.retry_delay_ms, 1000);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PROVIDER ENUM
// ═══════════════════════════════════════════════════════════════════════════════

mod provider_enum {
    use super::*;

    #[test]
    fn default_base_urls() {
        assert_eq!(
            Provider::DeepSeek.default_base_url(),
            "https://api.deepseek.com"
        );
        assert_eq!(
            Provider::OpenAI.default_base_url(),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            Provider::Anthropic.default_base_url(),
            "https://api.anthropic.com"
        );
        assert_eq!(
            Provider::Gemini.default_base_url(),
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert_eq!(
            Provider::Mimo.default_base_url(),
            "https://api.xiaomimimo.com/anthropic"
        );
    }

    #[test]
    fn default_models() {
        assert_eq!(Provider::DeepSeek.default_model(), "deepseek-chat");
        assert_eq!(Provider::OpenAI.default_model(), "gpt-4o");
        assert_eq!(
            Provider::Anthropic.default_model(),
            "claude-sonnet-4-20250514"
        );
        assert_eq!(Provider::Gemini.default_model(), "gemini-2.0-flash");
        assert_eq!(Provider::Mimo.default_model(), "mimo-v2.5-pro");
    }

    #[test]
    fn provider_is_copy() {
        let p = Provider::OpenAI;
        let p2 = p;
        assert_eq!(p, p2);
    }

    #[test]
    fn provider_serde_roundtrip() {
        let json = serde_json::to_string(&Provider::DeepSeek).unwrap();
        assert_eq!(json, r#""deepseek""#);
        let back: Provider = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Provider::DeepSeek);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MESSAGE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

mod message_types {
    use super::*;
    use agentix::request::ToolCall;

    #[test]
    fn user_message() {
        let m = user_msg("hello");
        assert!(matches!(m, Message::User(_)));
    }

    #[test]
    fn assistant_message() {
        let m = Message::Assistant {
            content: Some("hi".into()),
            reasoning: None,
            tool_calls: vec![],
            provider_data: None,
        };
        assert!(matches!(m, Message::Assistant { .. }));
    }

    #[test]
    fn tool_result_message() {
        let m = Message::ToolResult {
            call_id: "c1".into(),
            content: vec![agentix::request::Content::text("result")],
        };
        assert!(matches!(m, Message::ToolResult { .. }));
    }

    #[test]
    fn multi_turn_conversation() {
        let r = Request::new(Provider::DeepSeek, "k")
            .user("What is 2+2?")
            .message(Message::Assistant {
                content: Some("4".into()),
                reasoning: None,
                tool_calls: vec![],
                provider_data: None,
            })
            .user("And 3+3?");
        assert_eq!(r.messages.len(), 3);
    }

    #[test]
    fn tool_call_in_history() {
        let r = Request::new(Provider::DeepSeek, "k")
            .user("search X")
            .message(Message::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "search".into(),
                    arguments: r#"{"q":"X"}"#.into(),
                }],
                provider_data: None,
            })
            .message(Message::ToolResult {
                call_id: "call_1".into(),
                content: vec![agentix::request::Content::text("found X")],
            });
        assert_eq!(r.messages.len(), 3);
    }
}
