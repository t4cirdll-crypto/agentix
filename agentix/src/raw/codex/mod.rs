//! Raw `codex` provider.
//!
//! This drives `codex app-server --listen stdio://` directly. Unlike
//! `codex exec --json`, app-server emits token deltas and supports client-side
//! `dynamicTools`, so it can fit Agentix's stateless tool-call loop.

use std::process::Stdio;

use async_stream::stream;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, Command};
use tracing::warn;

use crate::config::AgentConfig;
use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::raw::shared::ToolDefinition;
use crate::request::{
    Content, ImageData, Message, ReasoningEffort, ResponseFormat, ToolCall, ToolChoice,
};
use crate::types::{CompleteResponse, FinishReason, ToolCallChunk, UsageStats};

fn image_url_for_turn(img: &crate::request::ImageContent) -> Result<String, ApiError> {
    match &img.data {
        ImageData::Base64(b) => Ok(format!("data:{};base64,{}", img.mime_type, b)),
        ImageData::Url(u) => Ok(u.clone()),
    }
}

fn user_parts_to_turn_input(
    parts: &[Content],
    reminder: Option<&str>,
) -> Result<Vec<Value>, ApiError> {
    let mut input = Vec::new();
    for part in parts {
        match part {
            Content::Text { text } => input.push(json!({"type": "text", "text": text})),
            Content::Image(img) => {
                input.push(json!({"type": "image", "url": image_url_for_turn(img)?}));
            }
            Content::Document(_) => {
                return Err(ApiError::Config(
                    "codex provider does not support document/PDF input; app-server UserInput has no document/input_file equivalent".into(),
                ));
            }
        }
    }
    if let Some(reminder) = reminder.filter(|s| !s.is_empty()) {
        input.push(json!({"type": "text", "text": reminder}));
    }
    Ok(input)
}

fn validate_supported_history(messages: &[Message]) -> Result<(), ApiError> {
    for msg in messages {
        match msg {
            Message::User(parts) => {
                for part in parts {
                    match part {
                        Content::Text { .. } => {}
                        Content::Image(img) => {
                            let _ = image_url_for_turn(img)?;
                        }
                        Content::Document(_) => {
                            return Err(ApiError::Config(
                                "codex provider does not support document/PDF input; app-server has no input_file equivalent".into(),
                            ));
                        }
                    }
                }
            }
            Message::ToolResult { content, .. } => {
                if content.iter().any(|c| !matches!(c, Content::Text { .. })) {
                    return Err(ApiError::Config(
                        "codex provider only supports text tool results; non-text tool output cannot be represented without loss".into(),
                    ));
                }
            }
            Message::Assistant { .. } => {}
        }
    }
    Ok(())
}

fn split_history_and_input(
    config: &AgentConfig,
    messages: &[Message],
) -> Result<(Vec<Value>, Vec<Value>), ApiError> {
    validate_supported_history(messages)?;

    let mut input = Vec::new();
    let last_idx = messages.len().saturating_sub(1);
    let mut history = Vec::new();
    for (idx, msg) in messages.iter().cloned().enumerate() {
        if idx == last_idx
            && let Message::User(parts) = &msg
        {
            input.extend(user_parts_to_turn_input(parts, config.reminder.as_deref())?);
        } else {
            history.push(msg);
        }
    }

    if input.is_empty() {
        if let Some(reminder) = config.reminder.as_ref().filter(|s| !s.is_empty()) {
            input.push(json!({"type": "text", "text": reminder}));
        } else {
            input.push(json!({"type": "text", "text": "Continue."}));
        }
    }

    let mut inject_config = config.clone();
    inject_config.reminder = None;
    let inject = crate::raw::openai::request::build_response_input_items(&inject_config, history)
        .into_iter()
        .map(normalize_inject_item)
        .collect();
    Ok((inject, input))
}

fn normalize_inject_item(mut item: Value) -> Value {
    let role = item
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if item.get("type").and_then(|v| v.as_str()) == Some("message")
        && item.get("content").is_some_and(|v| v.is_string())
        && let Some(text) = item
            .get("content")
            .and_then(|v| v.as_str())
            .map(str::to_string)
    {
        let part_type = if role == "assistant" {
            "output_text"
        } else {
            "input_text"
        };
        item["content"] = json!([{ "type": part_type, "text": text }]);
    }
    item
}

fn dynamic_tools(
    tools: &[ToolDefinition],
    tool_choice: Option<&ToolChoice>,
) -> Result<Vec<Value>, ApiError> {
    match tool_choice {
        Some(ToolChoice::None) => return Ok(Vec::new()),
        Some(ToolChoice::Auto) | None => {}
        Some(ToolChoice::Required) => {
            return Err(ApiError::Config(
                "codex provider cannot enforce tool_choice=required; app-server hard-codes Responses tool_choice=auto".into(),
            ));
        }
        Some(ToolChoice::Tool(name)) => {
            return Err(ApiError::Config(format!(
                "codex provider cannot force tool_choice for `{name}`; app-server hard-codes Responses tool_choice=auto"
            )));
        }
    }

    Ok(tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.function.name,
                "description": tool.function.description.clone().unwrap_or_default(),
                "inputSchema": tool.function.parameters,
            })
        })
        .collect::<Vec<_>>())
}

fn reasoning_effort(effort: Option<ReasoningEffort>) -> Option<&'static str> {
    match effort {
        Some(ReasoningEffort::None) => Some("minimal"),
        Some(ReasoningEffort::Minimal) => Some("minimal"),
        Some(ReasoningEffort::Low) => Some("low"),
        Some(ReasoningEffort::Medium) => Some("medium"),
        Some(ReasoningEffort::High) => Some("high"),
        Some(ReasoningEffort::XHigh) | Some(ReasoningEffort::Max) => Some("xhigh"),
        None => None,
    }
}

fn spawn_codex() -> Result<Child, ApiError> {
    let codex_bin = std::env::var("AGENTIX_CODEX_BIN").unwrap_or_else(|_| "codex".to_string());
    let mut cmd = if cfg!(windows) {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", &codex_bin, "app-server", "--listen", "stdio://"]);
        cmd
    } else {
        let mut cmd = Command::new(codex_bin);
        cmd.args(["app-server", "--listen", "stdio://"]);
        cmd
    };
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd.spawn()
        .map_err(|e| ApiError::Other(format!("spawn codex app-server: {e}")))
}

async fn write_json(stdin: &mut ChildStdin, value: &Value) -> Result<(), ApiError> {
    let mut line = value.to_string();
    line.push('\n');
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| ApiError::Other(format!("write codex app-server stdin: {e}")))
}

async fn read_json<R: AsyncBufRead + Unpin>(lines: &mut Lines<R>) -> Result<Value, ApiError> {
    loop {
        let line = lines
            .next_line()
            .await
            .map_err(|e| ApiError::Other(format!("read codex app-server stdout: {e}")))?
            .ok_or_else(|| ApiError::Other("codex app-server closed stdout".into()))?;
        if line.trim().is_empty() {
            continue;
        }
        return serde_json::from_str(&line)
            .map_err(|e| ApiError::Other(format!("decode codex app-server json: {e}: {line}")));
    }
}

async fn request<R: AsyncBufRead + Unpin>(
    stdin: &mut ChildStdin,
    lines: &mut Lines<R>,
    id: i64,
    method: &str,
    params: Value,
) -> Result<Value, ApiError> {
    write_json(
        stdin,
        &json!({"method": method, "id": id, "params": params}),
    )
    .await?;
    loop {
        let msg = read_json(lines).await?;
        if msg.get("id").and_then(|v| v.as_i64()) != Some(id) {
            continue;
        }
        if let Some(error) = msg.get("error") {
            return Err(ApiError::Other(format!(
                "codex app-server {method} error: {error}"
            )));
        }
        return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
    }
}

async fn initialize<R: AsyncBufRead + Unpin>(
    stdin: &mut ChildStdin,
    lines: &mut Lines<R>,
) -> Result<(), ApiError> {
    request(
        stdin,
        lines,
        1,
        "initialize",
        json!({
            "clientInfo": {"name": "agentix", "title": "Agentix", "version": env!("CARGO_PKG_VERSION")},
            "capabilities": {"experimentalApi": true}
        }),
    )
    .await?;
    write_json(stdin, &json!({"method": "initialized", "params": {}})).await
}

async fn start_thread<R: AsyncBufRead + Unpin>(
    stdin: &mut ChildStdin,
    lines: &mut Lines<R>,
    config: &AgentConfig,
    tools: &[ToolDefinition],
    tool_choice: Option<&ToolChoice>,
) -> Result<String, ApiError> {
    let cwd = std::env::current_dir()
        .map_err(|e| ApiError::Other(format!("current_dir: {e}")))?
        .to_string_lossy()
        .into_owned();
    let mut params = json!({
        "cwd": cwd,
        "model": config.model,
        "modelProvider": "openai",
        "approvalPolicy": "never",
        "sandbox": "read-only",
        "ephemeral": true,
        "environments": [],
        "experimentalRawEvents": true,
        "dynamicTools": dynamic_tools(tools, tool_choice)?,
    });
    if let Some(system) = config.system_prompt.as_ref().filter(|s| !s.is_empty()) {
        params["developerInstructions"] = Value::String(system.clone());
    }
    let result = request(stdin, lines, 2, "thread/start", params).await?;
    result
        .pointer("/thread/id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| ApiError::Other("codex thread/start did not return thread.id".into()))
}

fn validate_request_config(config: &AgentConfig) -> Result<(), ApiError> {
    if config.temperature.is_some() {
        return Err(ApiError::Config(
            "codex provider cannot map temperature; app-server turn/thread params expose no temperature field".into(),
        ));
    }
    if config.max_tokens.is_some() {
        return Err(ApiError::Config(
            "codex provider cannot map max_tokens/max_output_tokens; app-server turn/thread params expose no model output token limit".into(),
        ));
    }
    if !config.extra_body.is_empty() {
        return Err(ApiError::Config(
            "codex provider does not accept extra_body; unsupported fields would be ignored by app-server".into(),
        ));
    }
    if matches!(
        config.response_format.as_ref(),
        Some(ResponseFormat::JsonObject)
    ) {
        return Err(ApiError::Config(
            "codex provider cannot map JSON object mode exactly; use json_schema instead".into(),
        ));
    }
    if let Some(ResponseFormat::JsonSchema { strict: false, .. }) = config.response_format.as_ref()
    {
        return Err(ApiError::Config(
            "codex provider cannot map json_schema strict=false exactly; app-server outputSchema uses Codex's strict schema path".into(),
        ));
    }
    Ok(())
}

fn output_schema(config: &AgentConfig) -> Option<Value> {
    match &config.response_format {
        Some(ResponseFormat::JsonSchema { schema, .. }) => Some(schema.clone()),
        Some(ResponseFormat::Text) | Some(ResponseFormat::JsonObject) | None => None,
    }
}

fn parse_usage(value: &Value) -> UsageStats {
    let total = value.pointer("/params/tokenUsage/total").unwrap_or(value);
    let get = |key: &str| total.get(key).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let prompt = get("inputTokens");
    let completion = get("outputTokens");
    UsageStats {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: get("totalTokens").max(prompt + completion),
        cache_read_tokens: get("cachedInputTokens"),
        cache_creation_tokens: 0,
        reasoning_tokens: get("reasoningOutputTokens"),
    }
}

fn tool_call_from_params(params: &Value) -> Option<ToolCall> {
    Some(ToolCall {
        id: params.get("callId")?.as_str()?.to_string(),
        name: params.get("tool")?.as_str()?.to_string(),
        arguments: params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}))
            .to_string(),
    })
}

pub(crate) async fn stream_codex(
    _token: &str,
    _http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    tool_choice: Option<&ToolChoice>,
) -> Result<BoxStream<'static, LlmEvent>, ApiError> {
    validate_request_config(config)?;
    if messages.is_empty() {
        return Err(ApiError::Other(
            "codex needs at least one User or ToolResult message".into(),
        ));
    }

    let mut child = spawn_codex()?;
    if let Some(err) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                warn!(target: "codex_stderr", "{}", line);
            }
        });
    }
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ApiError::Other("codex app-server has no stdout".into()))?;
    let mut lines = BufReader::new(stdout).lines();
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ApiError::Other("codex app-server has no stdin".into()))?;

    initialize(&mut stdin, &mut lines).await?;
    let thread_id = start_thread(&mut stdin, &mut lines, config, tools, tool_choice).await?;

    let (inject_items, input) = split_history_and_input(config, messages)?;
    if !inject_items.is_empty() {
        request(
            &mut stdin,
            &mut lines,
            3,
            "thread/inject_items",
            json!({"threadId": thread_id, "items": inject_items}),
        )
        .await?;
    }
    let mut turn_params = json!({"threadId": thread_id, "input": input, "environments": []});
    if let Some(effort) = reasoning_effort(config.reasoning_effort) {
        turn_params["effort"] = Value::String(effort.to_string());
    }
    if let Some(schema) = output_schema(config) {
        turn_params["outputSchema"] = schema;
    }
    let result = request(&mut stdin, &mut lines, 4, "turn/start", turn_params).await?;
    let turn_id = result
        .pointer("/turn/id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(stream! {
        let child = child;
        let mut stdin = stdin;
        let mut lines = lines;
        let mut usage = UsageStats::default();
        let mut raw_items: Vec<Value> = Vec::new();

        loop {
            let msg = match read_json(&mut lines).await {
                Ok(msg) => msg,
                Err(e) => {
                    yield LlmEvent::Error(e.to_string());
                    break;
                }
            };
            let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
            match method {
                "item/agentMessage/delta" => {
                    if let Some(delta) = msg.pointer("/params/delta").and_then(|v| v.as_str())
                        && !delta.is_empty()
                    {
                        yield LlmEvent::Token(delta.to_string());
                    }
                }
                "item/tool/call" => {
                    if let Some(params) = msg.get("params")
                        && let Some(tc) = tool_call_from_params(params)
                    {
                        yield LlmEvent::ToolCallChunk(ToolCallChunk {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            delta: tc.arguments.clone(),
                            index: 0,
                        });
                        yield LlmEvent::ToolCall(tc);
                    }
                    let _ = write_json(&mut stdin, &json!({
                        "method": "turn/interrupt",
                        "id": 9000,
                        "params": {"threadId": thread_id, "turnId": turn_id},
                    })).await;
                    if !raw_items.is_empty() {
                        yield LlmEvent::AssistantState(json!({
                            "openai_responses_items": raw_items.clone()
                        }));
                    }
                    yield LlmEvent::Done;
                    break;
                }
                "rawResponseItem/completed" => {
                    if let Some(item) = msg.pointer("/params/item").cloned() {
                        raw_items.push(item);
                    }
                }
                "thread/tokenUsage/updated" => {
                    usage = parse_usage(&msg);
                    yield LlmEvent::Usage(usage.clone());
                }
                "turn/completed" => {
                    let status = msg.pointer("/params/turn/status").and_then(|v| v.as_str()).unwrap_or("");
                    if status == "failed" {
                        let err = msg.pointer("/params/turn/error").cloned().unwrap_or(Value::Null);
                        yield LlmEvent::Error(format!("codex turn failed: {err}"));
                    } else {
                        if usage.total_tokens != 0 {
                            yield LlmEvent::Usage(usage.clone());
                        }
                        if !raw_items.is_empty() {
                            yield LlmEvent::AssistantState(json!({
                                "openai_responses_items": raw_items.clone()
                            }));
                        }
                        yield LlmEvent::Done;
                    }
                    break;
                }
                _ => {
                    if let Some(error) = msg.get("error") {
                        yield LlmEvent::Error(format!("codex app-server error: {error}"));
                        break;
                    }
                }
            }
        }

        drop(stdin);
        drop(child);
    }
    .boxed())
}

pub(crate) async fn complete_codex(
    token: &str,
    http: &reqwest::Client,
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
    tool_choice: Option<&ToolChoice>,
) -> Result<CompleteResponse, ApiError> {
    let mut stream = stream_codex(token, http, config, messages, tools, tool_choice).await?;
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut provider_data = None;
    let mut usage = UsageStats::default();

    while let Some(ev) = stream.next().await {
        match ev {
            LlmEvent::Token(t) => content.push_str(&t),
            LlmEvent::ToolCall(tc) => tool_calls.push(tc),
            LlmEvent::AssistantState(data) => provider_data = Some(data),
            LlmEvent::Usage(u) => usage = u,
            LlmEvent::Error(e) => return Err(ApiError::Llm(e)),
            LlmEvent::Done => break,
            _ => {}
        }
    }

    Ok(CompleteResponse {
        content: if content.is_empty() {
            None
        } else {
            Some(content)
        },
        reasoning: None,
        tool_calls: tool_calls.clone(),
        provider_data,
        usage,
        finish_reason: if tool_calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCalls
        },
    })
}
