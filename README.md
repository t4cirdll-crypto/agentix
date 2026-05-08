# agentix

[![crates.io](https://img.shields.io/crates/v/agentix.svg)](https://crates.io/crates/agentix)
[![docs.rs](https://docs.rs/agentix/badge.svg)](https://docs.rs/agentix)
[![license](https://img.shields.io/crates/l/agentix.svg)](LICENSE)

Multi-provider LLM client for Rust: streaming, non-streaming, tool calls,
agent loops, MCP tools, structured output, multimodal input, and reasoning
state round-trip.

DeepSeek, OpenAI, Anthropic, Gemini, Kimi, GLM, MiniMax, Mimo, Grok, and
OpenRouter all use the same `Request` API.

---

## Quick Start

```rust
use agentix::{LlmEvent, Request};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let http = reqwest::Client::new();

    let mut stream = Request::deepseek(std::env::var("DEEPSEEK_API_KEY")?)
        .system_prompt("You are a helpful assistant.")
        .user("What is the capital of France?")
        .stream(&http)
        .await?;

    while let Some(event) = stream.next().await {
        match event {
            LlmEvent::Token(t) => print!("{t}"),
            LlmEvent::Done => break,
            LlmEvent::Error(e) => eprintln!("error: {e}"),
            _ => {}
        }
    }

    Ok(())
}
```

For one-shot requests:

```rust
let http = reqwest::Client::new();
let response = agentix::Request::openai(std::env::var("OPENAI_API_KEY")?)
    .user("Write a haiku about Rust.")
    .complete(&http)
    .await?;

println!("{}", response.content.unwrap_or_default());
```

---

## Installation

```toml
[dependencies]
agentix = "0.24"
```

Optional features:

```toml
# MCP client tools
agentix = { version = "0.24", features = ["mcp"] }

# Expose local tools as an MCP server
agentix = { version = "0.24", features = ["mcp-server"] }

# Use the local `claude -p` CLI as Provider::ClaudeCode
agentix = { version = "0.24", features = ["claude-code"] }

# Anthropic Messages-compatible HTTP server (POST /v1/messages, streaming SSE
# + non-streaming, fallback chain across upstreams). Library API is
# `agentix::server::AnthropicServer`.
agentix = { version = "0.24", features = ["server-anthropic"] }

# `agentix` CLI binary — Anthropic Messages proxy with fallback. Wraps
# server-anthropic with arg parsing for the headline use case.
agentix = { version = "0.24", features = ["cli"] }

# Compile-time gate for full request/response body logging
agentix = { version = "0.24", features = ["sensitive-logs"] }
```

The CLI binary takes one or more `-i <upstream>` flags (each opens a new
upstream in the fallback chain; trailing `--token / --model / --base-url`
flags bind to the most recent `-i`):

```bash
# Use Claude Code OAuth as primary, DeepSeek paid API as fallback.
agentix -i claude-code \
        -i https://api.deepseek.com/chat/completions --token $DEEPSEEK_API_KEY \
        --listen 127.0.0.1:7878
```

Then point any tool that speaks Anthropic Messages format at the proxy:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:7878 ANTHROPIC_API_KEY=any claude
```

---

## Design

`Request` is a value type. It contains provider, credentials, model, messages,
tools, and tuning knobs. Call `stream()` or `complete()` with a shared
`reqwest::Client`.

Agents are streams too. `agent()` emits token-level `AgentEvent`s across a full
LLM/tool loop; `agent_turns()` emits one `CompleteResponse` per LLM turn.

```rust
use agentix::{ToolBundle, agent_turns};

let text = agent_turns(ToolBundle::default(), http, request, history, Some(25_000))
    .last_content()
    .await;
```

Concurrency and pipelines are ordinary Rust:

```rust
use futures::future::join_all;

let answers = join_all(questions.into_iter().map(|question| {
    agentix::agent_turns(
        tools.clone(),
        http.clone(),
        request.clone(),
        vec![agentix::Message::User(vec![agentix::Content::text(question)])],
        None,
    )
    .last_content()
}))
.await;
```

---

## Comparison

This is a positioning snapshot, not a benchmark. External frameworks move
quickly; the agentix column tracks this repository's current behavior.

| | agentix | rig | llm-chain | LangGraph |
|---|---|---|---|---|
| Primary language | Rust | Rust | Rust | Python / JavaScript |
| Core abstraction | `Request` values and streams | Agents, providers, embeddings, vector stores | Chains / prompts | Stateful graph runtime |
| Agent loop | Built in: `agent()` / `agent_turns()` | Built in agent APIs | Manual / chain-oriented | Built in graph execution |
| Streaming text | Yes: `LlmEvent::Token` | Yes | Limited / provider-dependent | Yes |
| Streaming tool calls | Yes: chunks + completed calls | Provider/API-dependent | Limited | Yes, through LangGraph stream modes |
| Streaming tool progress | Yes: `ToolOutput::Progress` -> `AgentEvent::ToolProgress` | Custom app logic | Custom app logic | Yes, custom stream updates |
| Tool definition style | `#[tool]` on functions or impl blocks | Tool traits / derive macros | Chain/tool abstractions | LangChain tools or custom node logic |
| Tool grouping | `ToolBundle`, `+`, `+=`, `-`, `-=` | Agent/tool composition | Chain composition | Graph nodes / tool nodes |
| Multimodal input | Text, images, documents where provider supports them | Provider-dependent | Provider-dependent | Provider-dependent via model integrations |
| Structured output | JSON object + JSON Schema where provider supports it | Supported patterns vary by provider | Provider-dependent | Via model/tool integrations |
| Reasoning controls | Cross-provider `ReasoningEffort` | Provider-specific | Provider-specific | Provider/model-specific |
| Provider support | 10 HTTP providers + optional Claude Code CLI | Multiple native provider integrations | Older/smaller provider surface | Broad via LangChain ecosystem |
| MCP client tools | Optional `mcp` feature | Not core | Not core | Via integrations / custom nodes |
| MCP server | Optional `mcp-server` feature | Not core | Not core | Via integrations / deployment stack |

Why this table matters: agentix is intentionally not a graph framework. It keeps
provider calls, tool execution, and agent turns as regular Rust values and
streams, so complex workflows can be built with ordinary `async`, `Stream`, and
`Future` composition.

---

## Providers

Ten HTTP providers are built in. `Provider::ClaudeCode` is also available behind
the `claude-code` feature.

| Provider | Constructor | Default model | Default base URL | Wire format |
|---|---|---|---|---|
| DeepSeek | `Request::deepseek(key)` | `deepseek-chat` | `https://api.deepseek.com` | Chat Completions-compatible |
| OpenAI | `Request::openai(key)` | `gpt-4o` | `https://api.openai.com/v1` | Responses API |
| Anthropic | `Request::anthropic(key)` | `claude-sonnet-4-20250514` | `https://api.anthropic.com` | Messages API |
| Gemini | `Request::gemini(key)` | `gemini-2.0-flash` | `https://generativelanguage.googleapis.com/v1beta` | Gemini API |
| Kimi | `Request::kimi(key)` | `kimi-k2.5` | `https://api.moonshot.cn/v1` | Chat Completions-compatible |
| GLM | `Request::glm(key)` | `glm-5` | `https://open.bigmodel.cn/api/paas/v4` | Chat Completions-compatible |
| MiniMax | `Request::minimax(key)` | `MiniMax-M2.7` | `https://api.minimaxi.com/anthropic` | Anthropic-compatible |
| Mimo | `Request::mimo(key)` | `mimo-v2.5-pro` | `https://api.xiaomimimo.com/anthropic` | Anthropic-compatible |
| Grok | `Request::grok(key)` | `grok-4` | `https://api.x.ai/v1` | Chat Completions-compatible |
| OpenRouter | `Request::openrouter(key)` | `openrouter/auto` | `https://openrouter.ai/api/v1` | Chat Completions-compatible |

```rust
use agentix::{Provider, Request};

let req = Request::new(Provider::Mimo, std::env::var("MIMO_API_KEY")?)
    .model("mimo-v2.5")
    .user("Hello");
```

OpenAI is intentionally the official Responses API provider. For Azure, vLLM,
LocalAI, Ollama, llama.cpp server, or any endpoint that only speaks Chat
Completions, use `Provider::OpenRouter` with a custom base URL:

```rust
let req = Request::openrouter("local-key")
    .base_url("http://localhost:11434/v1")
    .model("llama3.1");
```

Mimo uses the documented `api-key: $MIMO_API_KEY` authentication header.

---

## Request API

```rust
use agentix::{Provider, ReasoningEffort, Request};

let req = Request::new(Provider::DeepSeek, "sk-...")
    .model("deepseek-v4-pro")
    .base_url("https://custom.api/v1")
    .system_prompt("You are helpful.")
    .reminder("<runtime_context>use current project settings</runtime_context>")
    .max_tokens(4096)
    .temperature(0.7)
    .reasoning_effort(ReasoningEffort::High)
    .retries(5, 2_000)
    .user("Hello")
    .tools(vec![]);
```

Useful builder methods:

- `model`, `base_url`, `system_prompt`, `reminder`
- `user`, `message`, `messages`
- `tools`
- `max_tokens`, `temperature`, `reasoning_effort`
- `text`, `json`, `json_schema`
- `extra_body` for provider-specific top-level JSON fields
- `retries(max, initial_delay_ms)`

`complete()` returns `CompleteResponse`:

```rust
let response = req.complete(&http).await?;
println!("text: {:?}", response.content);
println!("reasoning: {:?}", response.reasoning);
println!("tool calls: {:?}", response.tool_calls);
println!("usage: {:?}", response.usage);
println!("finish reason: {:?}", response.finish_reason);
```

---

## Streaming Events

`LlmEvent` is `#[non_exhaustive]`; include `_ => {}` in matches.

```rust
while let Some(event) = stream.next().await {
    match event {
        LlmEvent::Token(t) => print!("{t}"),
        LlmEvent::Reasoning(r) => eprint!("[reasoning] {r}"),
        LlmEvent::ToolCallChunk(chunk) => {
            eprintln!("tool args fragment: {}", chunk.delta);
        }
        LlmEvent::ToolCall(call) => {
            eprintln!("tool: {}({})", call.name, call.arguments);
        }
        LlmEvent::AssistantState(_) => {}
        LlmEvent::Usage(u) => eprintln!("tokens: {}", u.total_tokens),
        LlmEvent::Done => break,
        LlmEvent::Error(e) => eprintln!("error: {e}"),
        _ => {}
    }
}
```

Provider-specific reasoning state is captured as `AssistantState` and attached
to `Message::Assistant.provider_data` by the agent loop. User code usually does
not need to inspect it.

---

## Reasoning Control

`ReasoningEffort` is a single cross-provider knob:

```rust
use agentix::{ReasoningEffort, Request};

let req = Request::deepseek(key)
    .reasoning_effort(ReasoningEffort::Max)
    .user("Prove that there are infinitely many primes.");
```

| Variant | DeepSeek | Anthropic-compatible | OpenAI Responses | Gemini 3+ | Gemini 2.5 | OpenRouter | Other chat providers |
|---|---|---|---|---|---|---|---|
| `None` | disable thinking | disable thinking | omit reasoning | minimal floor | budget 0 | `none` | ignored |
| `Minimal` | high | low | minimal | minimal | 512 | minimal | ignored |
| `Low` | high | low | low | low | 1024 | low | ignored |
| `Medium` | high | medium | medium | medium | 4096 | medium | ignored |
| `High` | high | high | high | high | 8192 | high | ignored |
| `XHigh` | max | xhigh | xhigh | high | 16384 | xhigh | ignored |
| `Max` | max | max | high | high | 24576 | max | ignored |
| unset | provider default | provider default | omitted | omitted | omitted | omitted | omitted |

Notes:

- `ReasoningEffort::None` is different from leaving the field unset. `None`
  explicitly disables thinking where the provider supports that toggle.
- DeepSeek drops sampling parameters such as `temperature` while thinking is
  enabled, because its API rejects that combination.
- Thinking/tool-call state is automatically round-tripped for Anthropic-compatible
  providers, OpenAI Responses, Gemini, and OpenRouter.

See [examples/11_reasoning.rs](agentix/examples/11_reasoning.rs).

---

## Messages And Multimodal Input

User messages are `Vec<Content>`:

```rust
use agentix::{Content, DocumentContent, DocumentData, ImageContent, ImageData, Request};

let req = Request::anthropic(key).message(agentix::Message::User(vec![
    Content::text("Summarize this document and image."),
    Content::Document(DocumentContent {
        data: DocumentData::Base64(pdf_base64),
        mime_type: "application/pdf".into(),
        filename: Some("paper.pdf".into()),
    }),
    Content::Image(ImageContent {
        data: ImageData::Url("https://example.com/chart.png".into()),
        mime_type: "image/png".into(),
    }),
]));
```

Document support:

- Anthropic-compatible providers emit `document` blocks.
- OpenAI Responses emits `input_file`.
- Gemini emits `inline_data` or `file_data`.
- OpenRouter emits file parts for providers/plugins that support them.
- DeepSeek, Grok, GLM, and Kimi silently drop document parts.

Images are supported by providers whose wire format accepts them. If a provider
does not accept a content type, agentix drops or degrades the part rather than
inventing an incompatible schema.

---

## Tools

Use `#[tool]` on standalone functions or an `impl agentix::Tool` block.
Doc comments become tool and parameter descriptions.

```rust
use agentix::{ToolBundle, tool};

/// Add two numbers.
/// a: first number
/// b: second number
#[tool]
async fn add(a: i64, b: i64) -> i64 {
    a + b
}

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Divide a by b.
    /// a: numerator
    /// b: denominator
    async fn divide(&self, a: f64, b: f64) -> Result<f64, String> {
        if b == 0.0 {
            Err("division by zero".into())
        } else {
            Ok(a / b)
        }
    }
}

let tools = ToolBundle::default() + add + Calculator;
```

Run a full agent loop:

```rust
use agentix::{AgentEvent, Message, Request, ToolBundle};
use futures::StreamExt;

let http = reqwest::Client::new();
let request = Request::deepseek(std::env::var("DEEPSEEK_API_KEY")?)
    .system_prompt("Use tools for arithmetic.");
let history = vec![Message::User(vec![agentix::Content::text("What is 12 / 3?")])];

let mut stream = agentix::agent(ToolBundle::default() + Calculator, http, request, history, None);

while let Some(event) = stream.next().await {
    match event {
        AgentEvent::Token(t) => print!("{t}"),
        AgentEvent::ToolCallStart(call) => eprintln!("tool: {}", call.name),
        AgentEvent::ToolProgress { progress, .. } => eprintln!("progress: {progress}"),
        AgentEvent::ToolResult { name, content, .. } => eprintln!("{name}: {content:?}"),
        AgentEvent::Done(usage) => eprintln!("tokens: {}", usage.total_tokens),
        AgentEvent::Error(e) => eprintln!("error: {e}"),
        _ => {}
    }
}
```

Streaming tools can yield progress before their final result:

```rust
use agentix::{ToolOutput, tool};

struct Jobs;

#[tool]
impl agentix::Tool for Jobs {
    /// Run a job.
    /// steps: number of steps
    #[streaming]
    fn run_job(&self, steps: u32) {
        async_stream::stream! {
            for step in 1..=steps {
                yield ToolOutput::Progress(format!("{step}/{steps}"));
            }
            yield ToolOutput::Result(vec![agentix::Content::text("done")]);
        }
    }
}
```

`ToolBundle` supports `new`, `with`, `push`, `remove`, `+`, `+=`, `-`, and `-=`.

---

## MCP

MCP client tools require the `mcp` feature:

```rust
use agentix::{McpTool, ToolBundle};
use std::time::Duration;

let playwright = McpTool::stdio("npx", &["-y", "@playwright/mcp"])
    .await?
    .with_timeout(Duration::from_secs(60))
    .with_output_limits(20_000, 20);

let tools = ToolBundle::default() + playwright;
```

The `mcp-server` feature exposes local `ToolBundle`s as MCP services. See
[examples/06_mcp_server.rs](agentix/examples/06_mcp_server.rs).

---

## Structured Output

For JSON object mode:

```rust
let response = Request::openai(key)
    .system_prompt("Return JSON only.")
    .user("Return {\"ok\": true}.")
    .json()
    .complete(&http)
    .await?;
```

For JSON Schema mode:

```rust
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Review {
    rating: f32,
    summary: String,
    pros: Vec<String>,
}

let schema = serde_json::to_value(schemars::schema_for!(Review))?;
let response = Request::openai(key)
    .system_prompt("You are a film critic.")
    .user("Review Inception.")
    .json_schema("review", schema, true)
    .complete(&http)
    .await?;

let review: Review = response.json()?;
```

Provider behavior:

- OpenAI Responses supports text, JSON object, and JSON Schema.
- Gemini supports JSON object and JSON Schema through generation config.
- DeepSeek degrades JSON Schema to JSON object with a warning.
- Grok, GLM, Kimi, and OpenRouter pass compatible `response_format` fields.
- Anthropic-compatible providers ignore `response_format`; use prompting or
  tools for strict structure.

See [examples/08_structured_output.rs](agentix/examples/08_structured_output.rs).

---

## Claude Code

With the `claude-code` feature, `Provider::ClaudeCode` runs the local
`claude -p` CLI and lets agentix keep control of the LLM/tool loop. Auth comes
from the Claude CLI OAuth session.

```toml
agentix = { version = "0.24", features = ["claude-code"] }
```

```rust
use agentix::{AgentEvent, Content, Message, Request, agent, tool};
use futures::StreamExt;

struct Calculator;

#[tool]
impl agentix::Tool for Calculator {
    /// Add two numbers.
    /// a: first number
    /// b: second number
    async fn add(&self, a: f64, b: f64) -> f64 {
        a + b
    }
}

let http = reqwest::Client::new();
let request = Request::claude_code()
    .model("sonnet")
    .system_prompt("Always use tools for arithmetic.");
let history = vec![Message::User(vec![Content::text("What is 123 + 456?")])];

let mut stream = agent(Calculator, http, request, history, None);
while let Some(event) = stream.next().await {
    match event {
        AgentEvent::Token(t) => print!("{t}"),
        AgentEvent::Done(usage) => eprintln!("tokens: {}", usage.total_tokens),
        _ => {}
    }
}
```

See [examples/10_claude_code.rs](agentix/examples/10_claude_code.rs).

---

## Sensitive Logging

Full request bodies, response bodies, streaming chunks, and MCP raw request
bodies are sensitive and disabled by default. To enable them, opt in at compile
time and runtime:

```bash
AGENTIX_LOG_BODIES=1 cargo run --features sensitive-logs
```

If either gate is missing, full bodies are not logged.

---

## Examples

- [01_streaming.rs](agentix/examples/01_streaming.rs): streaming tokens
- [02_completion.rs](agentix/examples/02_completion.rs): non-streaming completion
- [03_conversation.rs](agentix/examples/03_conversation.rs): conversation state
- [04_tools.rs](agentix/examples/04_tools.rs): tool definitions
- [05_mcp_client.rs](agentix/examples/05_mcp_client.rs): MCP client tools
- [06_mcp_server.rs](agentix/examples/06_mcp_server.rs): MCP server
- [07_agent.rs](agentix/examples/07_agent.rs): agent loop
- [08_structured_output.rs](agentix/examples/08_structured_output.rs): JSON schema output
- [09_deep_research.rs](agentix/examples/09_deep_research.rs): multi-step research flow
- [10_claude_code.rs](agentix/examples/10_claude_code.rs): Claude Code provider
- [11_reasoning.rs](agentix/examples/11_reasoning.rs): reasoning effort comparison

---

## License

MIT OR Apache-2.0
