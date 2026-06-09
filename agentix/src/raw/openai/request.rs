//! Request wire format for OpenAI's Responses API (`POST /v1/responses`).
//!
//! Stateless mode only: we always send `store: false`, never
//! `previous_response_id`. Encrypted reasoning is requested via
//! `include: ["reasoning.encrypted_content"]` on reasoning models (o-series,
//! gpt-5*) so multi-turn tool loops can round-trip the model's hidden chain
//! of thought via [`crate::request::Message::Assistant::provider_data`]. On
//! non-reasoning models (gpt-4o, gpt-4.1) the API rejects that include value
//! with `"Encrypted content is not supported with this model"`, so we omit it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{DocumentData, ImageData, Message, ReasoningEffort, UserContent};

// ── Request ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct Request {
    pub model: String,
    pub input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    pub store: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<&'static str>,
    #[serde(flatten)]
    pub extra_body: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct ReasoningConfig {
    pub effort: &'static str,
}

/// Structured-output config — `text.format` replaces Chat Completions'
/// `response_format`.
#[derive(Debug, Serialize)]
pub struct TextConfig {
    pub format: TextFormat,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: bool,
    },
}

// ── Input items ───────────────────────────────────────────────────────────────

/// One element of the `input[]` array.
///
/// `#[serde(untagged)]` — the inner enums are each `#[serde(tag = "type")]`
/// so each variant lands on a known discriminator. Using `untagged` at this
/// layer lets us emit provider_data items (opaque `serde_json::Value`) in the
/// same stream without re-parsing them into our typed enum.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum InputItem {
    Typed(TypedItem),
    /// Raw JSON from `provider_data.openai_responses_items` — emitted verbatim
    /// to preserve `encrypted_content`, `id` fields, and interleaved ordering
    /// of reasoning/function_call items that the server needs to validate.
    Raw(Value),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedItem {
    Message {
        role: &'static str,
        content: MessageContent,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    InputText {
        text: String,
    },
    InputImage {
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Document (PDF etc.). Either inline base64 via `file_data` (must be a
    /// data URL) or a public `file_url`. `filename` is required alongside
    /// `file_data`.
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

// ── Tools ─────────────────────────────────────────────────────────────────────

/// Flat-typed function tool shape. Differs from Chat Completions'
/// `{type: "function", function: {name, description, parameters}}` — here all
/// fields sit at the top level of the tool entry.
#[derive(Debug, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    /// Force a specific function. Flat shape — NOT wrapped under `function`.
    Function {
        name: String,
    },
}

// ── Deserialization helpers for provider_data → Raw input items ───────────────
//
// We only need Deserialize on the public Message view for provider_data JSON
// extraction; the rest of the request struct is Serialize-only.

#[derive(Debug, Deserialize)]
pub(crate) struct RawItems {
    pub openai_responses_items: Vec<Value>,
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub(crate) fn build_responses_request(
    config: &AgentConfig,
    history: Vec<Message>,
    tools: &[ToolDefinition],
    tool_choice: Option<ToolChoice>,
    stream: bool,
) -> Request {
    let mut input: Vec<InputItem> = Vec::new();

    for m in history {
        match m {
            Message::User(parts) => {
                input.push(InputItem::Typed(TypedItem::Message {
                    role: "user",
                    content: user_content_from_parts(parts),
                }));
            }
            Message::Assistant {
                content,
                tool_calls,
                provider_data,
                ..
            } => {
                // Prefer the opaque wire form captured from a previous turn —
                // it carries reasoning items with their `encrypted_content`
                // and preserves the relative order of reasoning / function_call
                // items that the server validates against.
                if let Some(items) = provider_data
                    .as_ref()
                    .and_then(|v| serde_json::from_value::<RawItems>(v.clone()).ok())
                    .filter(|r| !r.openai_responses_items.is_empty())
                {
                    for item in items.openai_responses_items {
                        input.push(InputItem::Raw(item));
                    }
                    continue;
                }
                if let Some(text) = content.filter(|s| !s.is_empty()) {
                    input.push(InputItem::Typed(TypedItem::Message {
                        role: "assistant",
                        content: MessageContent::Text(text),
                    }));
                }
                for tc in tool_calls {
                    input.push(InputItem::Typed(TypedItem::FunctionCall {
                        call_id: tc.id,
                        name: tc.name,
                        arguments: tc.arguments,
                    }));
                }
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                // Responses API wants a plain string output. Concatenate text
                // parts; images and documents are not accepted in
                // `function_call_output`, so drop them.
                let output: String = content
                    .iter()
                    .filter_map(|p| match p {
                        Content::Text { text } => Some(text.as_str()),
                        Content::Image(_) | Content::Document(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                input.push(InputItem::Typed(TypedItem::FunctionCallOutput {
                    call_id,
                    output,
                }));
            }
        }
    }

    let tools_opt = if tools.is_empty() {
        None
    } else {
        Some(tools.iter().map(openai_tool_from_shared).collect())
    };

    let extra = if config.extra_body.is_empty() {
        None
    } else {
        Some(config.extra_body.clone())
    };

    let include = if should_include_encrypted(config) {
        vec!["reasoning.encrypted_content"]
    } else {
        Vec::new()
    };

    Request {
        model: config.model.clone(),
        input,
        instructions: config.system_prompt.clone().filter(|s| !s.is_empty()),
        tools: tools_opt,
        tool_choice,
        stream: Some(stream),
        temperature: config.temperature,
        max_output_tokens: config.max_tokens,
        reasoning: openai_reasoning_config(config.reasoning_effort),
        text: text_config_from(config.response_format.as_ref()),
        store: false,
        include,
        extra_body: extra,
    }
}

pub(crate) fn build_response_input_items(
    config: &AgentConfig,
    history: Vec<Message>,
) -> Vec<Value> {
    let req = build_responses_request(config, history, &[], None, false);
    serde_json::to_value(req)
        .ok()
        .and_then(|v| v.get("input").cloned())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default()
}

/// Responses API rejects `include: ["reasoning.encrypted_content"]` on
/// non-reasoning models (gpt-4o, gpt-4.1) with `"Encrypted content is not
/// supported with this model"`. Include only when we know we're talking to
/// a reasoning model, which we detect via an explicit non-None
/// `reasoning_effort` (user opt-in) or a known reasoning-family prefix.
fn should_include_encrypted(config: &AgentConfig) -> bool {
    match config.reasoning_effort {
        // User explicitly disabled thinking — don't ask for encrypted content.
        Some(ReasoningEffort::None) => false,
        // Any non-None effort is a user opt-in; trust it even on custom
        // model names (third-party reasoning deployments).
        Some(_) => true,
        // Default: decide by model family.
        None => is_reasoning_model(&config.model),
    }
}

fn is_reasoning_model(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    // o-series (o1, o3, o4-mini). OpenAI skipped o2.
    if m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") {
        return true;
    }
    // gpt-5 family (all variants reason).
    m.starts_with("gpt-5")
}

// ── Translation helpers ───────────────────────────────────────────────────────

fn user_content_from_parts(parts: Vec<UserContent>) -> MessageContent {
    // Single plain-text part → string shorthand. Anything richer → parts array.
    if parts.len() == 1
        && let Some(UserContent::Text { text }) = parts.first()
    {
        return MessageContent::Text(text.clone());
    }
    let blocks = parts
        .into_iter()
        .map(|p| match p {
            UserContent::Text { text } => ContentPart::InputText { text },
            UserContent::Image(img) => {
                let image_url = match img.data {
                    ImageData::Url(u) => u,
                    ImageData::Base64(b) => format!("data:{};base64,{}", img.mime_type, b),
                };
                ContentPart::InputImage {
                    image_url,
                    detail: None,
                }
            }
            UserContent::Document(doc) => match doc.data {
                DocumentData::Url(u) => ContentPart::InputFile {
                    file_data: None,
                    file_url: Some(u),
                    filename: doc.filename,
                },
                DocumentData::Base64(b) => ContentPart::InputFile {
                    file_data: Some(format!("data:{};base64,{}", doc.mime_type, b)),
                    file_url: None,
                    // OpenAI requires a filename alongside `file_data`. Fall
                    // back to a generic PDF placeholder if none was supplied.
                    filename: Some(
                        doc.filename
                            .unwrap_or_else(|| default_filename(&doc.mime_type)),
                    ),
                },
            },
        })
        .collect();
    MessageContent::Parts(blocks)
}

fn default_filename(mime: &str) -> String {
    match mime {
        "application/pdf" => "document.pdf".into(),
        _ => "document".into(),
    }
}

fn openai_tool_from_shared(t: &ToolDefinition) -> Tool {
    Tool {
        kind: "function",
        name: t.function.name.clone(),
        description: t.function.description.clone(),
        parameters: t.function.parameters.clone(),
        // Responses API makes function tools strict-by-default for better
        // reasoning perf. Users who want loose schemas can bypass via
        // extra_body or route through Provider::OpenRouter.
        strict: false,
    }
}

/// Map cross-provider [`ReasoningEffort`] to the Responses API's
/// `reasoning.effort` enum. `None` / `Max` collapse where the API doesn't
/// have a matching level; `ReasoningEffort::None` means "omit the reasoning
/// config" (non-reasoning models reject the parameter).
pub(crate) fn openai_reasoning_config(effort: Option<ReasoningEffort>) -> Option<ReasoningConfig> {
    let e = effort?;
    let wire = match e {
        ReasoningEffort::None => return None,
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
        ReasoningEffort::Max => "high",
    };
    Some(ReasoningConfig { effort: wire })
}

fn text_config_from(rf: Option<&crate::request::ResponseFormat>) -> Option<TextConfig> {
    use crate::request::ResponseFormat;
    let rf = rf?;
    let format = match rf {
        ResponseFormat::Text => TextFormat::Text,
        ResponseFormat::JsonObject => TextFormat::JsonObject,
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => TextFormat::JsonSchema {
            name: name.clone(),
            schema: schema.clone(),
            strict: *strict,
        },
    };
    Some(TextConfig { format })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Content, Message, ToolCall};

    fn cfg() -> AgentConfig {
        AgentConfig {
            model: "gpt-5".into(),
            ..Default::default()
        }
    }

    #[test]
    fn system_prompt_is_instructions_not_message_item() {
        let mut c = cfg();
        c.system_prompt = Some("You are helpful.".into());
        let req = build_responses_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["instructions"], "You are helpful.");
        // input[] must not carry a role:system item
        let input = json["input"].as_array().unwrap();
        for item in input {
            assert_ne!(item["role"], "system");
        }
    }

    #[test]
    fn store_false_and_include_encrypted_on_reasoning_model() {
        // gpt-5 is a reasoning model → include without needing explicit effort.
        let req = build_responses_request(&cfg(), vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["store"], false);
        assert_eq!(
            json["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn include_omitted_on_non_reasoning_model() {
        // gpt-4o rejects `include: ["reasoning.encrypted_content"]` — must omit.
        let mut c = cfg();
        c.model = "gpt-4o".into();
        let req = build_responses_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        // include is serialized only when non-empty (skip_serializing_if).
        assert!(json.get("include").is_none());
    }

    #[test]
    fn include_opt_in_via_explicit_effort_on_unknown_model() {
        // Custom model name + explicit reasoning effort → trust the user
        // and include (e.g. third-party reasoning deployment).
        let mut c = cfg();
        c.model = "custom-reasoner-v1".into();
        c.reasoning_effort = Some(ReasoningEffort::High);
        let req = build_responses_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn include_omitted_when_effort_is_none() {
        // User explicitly disabled reasoning → omit include even on gpt-5,
        // mirroring the reasoning config omission.
        let mut c = cfg();
        c.reasoning_effort = Some(ReasoningEffort::None);
        let req = build_responses_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("include").is_none());
    }

    #[test]
    fn user_text_to_string_shorthand() {
        let req = build_responses_request(
            &cfg(),
            vec![Message::User(vec![Content::text("hi")])],
            &[],
            None,
            false,
        );
        let json = serde_json::to_value(&req).unwrap();
        let items = json["input"].as_array().unwrap();
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[0]["role"], "user");
        assert_eq!(items[0]["content"], "hi");
    }

    #[test]
    fn reasoning_effort_maps_to_reasoning_config() {
        let mut c = cfg();
        c.reasoning_effort = Some(ReasoningEffort::High);
        let req = build_responses_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["reasoning"]["effort"], "high");
    }

    #[test]
    fn reasoning_effort_none_omits_config() {
        let mut c = cfg();
        c.reasoning_effort = Some(ReasoningEffort::None);
        let req = build_responses_request(&c, vec![], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json["reasoning"].is_null());
    }

    #[test]
    fn provider_data_items_splice_in_order() {
        // Interleaved reasoning → function_call → reasoning → function_call.
        // Responses API validates this ordering; our splice must preserve it.
        let pd = serde_json::json!({
            "openai_responses_items": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc_1"},
                {"type": "function_call", "call_id": "call_1", "name": "x", "arguments": "{}", "id": "fc_1"},
                {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc_2"},
                {"type": "function_call", "call_id": "call_2", "name": "x", "arguments": "{}", "id": "fc_2"},
            ]
        });
        let history = vec![
            Message::User(vec![Content::text("hi")]),
            Message::Assistant {
                content: Some("ignored".into()),
                reasoning: Some("ignored".into()),
                tool_calls: vec![ToolCall {
                    id: "ignored".into(),
                    name: "ignored".into(),
                    arguments: "{}".into(),
                }],
                provider_data: Some(pd),
            },
        ];
        let req = build_responses_request(&cfg(), history, &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        let items = json["input"].as_array().unwrap();
        // [0] = user message, [1..=4] = the 4 spliced items verbatim
        assert_eq!(items.len(), 5);
        assert_eq!(items[1]["type"], "reasoning");
        assert_eq!(items[1]["encrypted_content"], "enc_1");
        assert_eq!(items[2]["type"], "function_call");
        assert_eq!(items[2]["call_id"], "call_1");
        assert_eq!(items[3]["type"], "reasoning");
        assert_eq!(items[3]["encrypted_content"], "enc_2");
        assert_eq!(items[4]["type"], "function_call");
        assert_eq!(items[4]["call_id"], "call_2");
    }

    #[test]
    fn document_base64_emits_input_file_with_filename() {
        use crate::request::{DocumentContent, DocumentData, UserContent};
        let parts = vec![
            UserContent::Text {
                text: "please read".into(),
            },
            UserContent::Document(DocumentContent {
                data: DocumentData::Base64("UERGZmFrZQ==".into()),
                mime_type: "application/pdf".into(),
                filename: Some("spec.pdf".into()),
            }),
        ];
        let req = build_responses_request(&cfg(), vec![Message::User(parts)], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        let parts = json["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[1]["type"], "input_file");
        assert_eq!(parts[1]["filename"], "spec.pdf");
        assert_eq!(
            parts[1]["file_data"],
            "data:application/pdf;base64,UERGZmFrZQ=="
        );
        assert!(parts[1]["file_url"].is_null());
    }

    #[test]
    fn document_url_emits_input_file_with_file_url() {
        use crate::request::{DocumentContent, DocumentData, UserContent};
        let parts = vec![
            UserContent::Text {
                text: "read this".into(),
            },
            UserContent::Document(DocumentContent {
                data: DocumentData::Url("https://example.com/doc.pdf".into()),
                mime_type: "application/pdf".into(),
                filename: None,
            }),
        ];
        let req = build_responses_request(&cfg(), vec![Message::User(parts)], &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        let parts = json["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[1]["type"], "input_file");
        assert_eq!(parts[1]["file_url"], "https://example.com/doc.pdf");
        assert!(parts[1]["file_data"].is_null());
    }

    #[test]
    fn tool_result_becomes_function_call_output() {
        let history = vec![Message::ToolResult {
            call_id: "call_abc".into(),
            content: vec![Content::text("42")],
        }];
        let req = build_responses_request(&cfg(), history, &[], None, false);
        let json = serde_json::to_value(&req).unwrap();
        let items = json["input"].as_array().unwrap();
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[0]["call_id"], "call_abc");
        assert_eq!(items[0]["output"], "42");
    }

    #[test]
    fn tool_definition_emits_flat_shape() {
        let tools = vec![ToolDefinition::function(
            crate::raw::shared::FunctionDefinition {
                name: "get_weather".into(),
                description: Some("fetch weather".into()),
                parameters: serde_json::json!({"type": "object"}),
                strict: None,
            },
        )];
        let req = build_responses_request(&cfg(), vec![], &tools, None, false);
        let json = serde_json::to_value(&req).unwrap();
        let t = &json["tools"][0];
        // Flat shape — NOT nested under "function".
        assert_eq!(t["type"], "function");
        assert_eq!(t["name"], "get_weather");
        assert_eq!(t["description"], "fetch weather");
        assert!(t["parameters"].is_object());
        assert!(t["function"].is_null());
    }
}
