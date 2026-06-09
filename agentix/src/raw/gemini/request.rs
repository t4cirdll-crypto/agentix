use serde::Serialize;
use serde_json::Value;

use crate::config::AgentConfig;
use crate::raw::shared::ToolDefinition;
use crate::request::{DocumentData, ImageData, Message, ReasoningEffort, UserContent};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Request {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTools>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize)]
pub struct SystemInstruction {
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
pub struct Content {
    pub role: &'static str,
    pub parts: Vec<Part>,
}

#[derive(Debug)]
pub enum Part {
    Text(String),
    InlineData(Blob),
    /// Public URL pointer (PDFs, videos, etc.). Maps to Gemini's `file_data`
    /// part shape: `{file_data: {mime_type, file_uri}}`.
    FileData(FileData),
    FunctionCall(FunctionCall),
    FunctionResponse(FunctionResponse),
    /// Raw opaque JSON — emitted verbatim to preserve a part captured from
    /// a previous turn (with its `thoughtSignature`). Used when splicing
    /// `provider_data.gemini_parts` back into contents[]. Gemini 3 rejects
    /// turns where the signature is missing or reordered.
    Raw(Value),
}

impl serde::Serialize for Part {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            Part::Raw(v) => v.serialize(s),
            _ => {
                let mut map = s.serialize_map(None)?;
                match self {
                    Part::Text(t) => {
                        map.serialize_entry("text", t)?;
                    }
                    Part::InlineData(b) => {
                        map.serialize_entry("inline_data", b)?;
                    }
                    Part::FileData(fd) => {
                        map.serialize_entry("file_data", fd)?;
                    }
                    Part::FunctionCall(fc) => {
                        map.serialize_entry("function_call", fc)?;
                    }
                    Part::FunctionResponse(fr) => {
                        map.serialize_entry("function_response", fr)?;
                    }
                    Part::Raw(_) => unreachable!(),
                }
                map.end()
            }
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Blob {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

#[derive(Debug, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Serialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Serialize)]
pub struct GeminiTools {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Serialize)]
pub struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize)]
pub struct FunctionCallingConfig {
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
}

#[derive(Debug, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ThinkingConfig {
    /// Gemini 3 models accept a qualitative level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<&'static str>,
    /// Gemini 2.5 models want a numeric token budget (`-1` = dynamic, `0` =
    /// disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
    /// Ask the model to return `thought: true` parts with summarized
    /// reasoning text. We set this whenever a thinking config is emitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
}

pub(crate) fn build_gemini_request(
    config: &AgentConfig,
    messages: &[Message],
    tools: &[ToolDefinition],
) -> Request {
    let system_instruction = config
        .system_prompt
        .as_ref()
        .filter(|s| !s.is_empty())
        .map(|s| SystemInstruction {
            parts: vec![Part::Text(s.clone())],
        });

    let mut contents: Vec<Content> = Vec::new();
    let mut pending_fn_responses: Vec<Part> = Vec::new();

    for msg in messages {
        match msg {
            Message::User(parts) => {
                if !pending_fn_responses.is_empty() {
                    contents.push(Content {
                        role: "user",
                        parts: std::mem::take(&mut pending_fn_responses),
                    });
                }
                contents.push(Content {
                    role: "user",
                    parts: parts
                        .iter()
                        .map(|p| match p {
                            UserContent::Text { text: t } => Part::Text(t.clone()),
                            UserContent::Image(img) => Part::InlineData(Blob {
                                mime_type: img.mime_type.clone(),
                                data: match &img.data {
                                    ImageData::Base64(b) => b.clone(),
                                    ImageData::Url(u) => u.clone(),
                                },
                            }),
                            UserContent::Document(doc) => match &doc.data {
                                DocumentData::Base64(b) => Part::InlineData(Blob {
                                    mime_type: doc.mime_type.clone(),
                                    data: b.clone(),
                                }),
                                DocumentData::Url(u) => Part::FileData(FileData {
                                    mime_type: doc.mime_type.clone(),
                                    file_uri: u.clone(),
                                }),
                            },
                        })
                        .collect(),
                });
            }
            Message::Assistant {
                content,
                tool_calls,
                provider_data,
                ..
            } => {
                // Prefer the raw wire form from a previous turn — it carries
                // `thoughtSignature` on function-call parts in their original
                // relative position. Gemini 3 400s if the signature is missing
                // or if the ordering between thought + functionCall parts has
                // been altered.
                if let Some(raw_parts) = provider_data
                    .as_ref()
                    .and_then(|v| v.get("gemini_parts"))
                    .and_then(|v| v.as_array())
                    .filter(|a| !a.is_empty())
                {
                    let parts: Vec<Part> = raw_parts.iter().map(|p| Part::Raw(p.clone())).collect();
                    contents.push(Content {
                        role: "model",
                        parts,
                    });
                } else {
                    let mut parts: Vec<Part> = Vec::new();
                    if let Some(t) = content
                        && !t.is_empty()
                    {
                        parts.push(Part::Text(t.clone()));
                    }
                    for tc in tool_calls {
                        let args = serde_json::from_str(&tc.arguments).unwrap_or(Value::Null);
                        parts.push(Part::FunctionCall(FunctionCall {
                            name: tc.name.clone(),
                            args,
                        }));
                    }
                    if !parts.is_empty() {
                        contents.push(Content {
                            role: "model",
                            parts,
                        });
                    }
                }
            }
            Message::ToolResult { call_id, content } => {
                use crate::request::Content;
                let text = content
                    .iter()
                    .filter_map(|p| {
                        if let Content::Text { text } = p {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                pending_fn_responses.push(Part::FunctionResponse(FunctionResponse {
                    name: call_id.clone(),
                    response: serde_json::json!({ "result": text }),
                }));
                // Append image parts alongside the function response in the same content block.
                for p in content {
                    if let Content::Image(img) = p {
                        let data = match &img.data {
                            ImageData::Base64(b) => b.clone(),
                            ImageData::Url(u) => u.clone(),
                        };
                        pending_fn_responses.push(Part::InlineData(Blob {
                            mime_type: img.mime_type.clone(),
                            data,
                        }));
                    }
                }
            }
        }
    }
    if !pending_fn_responses.is_empty() {
        contents.push(Content {
            role: "user",
            parts: pending_fn_responses,
        });
    }
    let gemini_tools = if tools.is_empty() {
        None
    } else {
        Some(vec![GeminiTools {
            function_declarations: tools
                .iter()
                .map(|t| FunctionDeclaration {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: sanitize_schema_for_gemini(t.function.parameters.clone()),
                })
                .collect(),
        }])
    };

    let tool_config = if tools.is_empty() {
        None
    } else {
        Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                mode: "AUTO",
                allowed_function_names: None,
            },
        })
    };

    let (response_mime_type, response_schema) = match &config.response_format {
        Some(crate::request::ResponseFormat::JsonObject) => (Some("application/json"), None),
        Some(crate::request::ResponseFormat::JsonSchema { schema, .. }) => {
            (Some("application/json"), Some(schema.clone()))
        }
        _ => (None, None),
    };
    let thinking_config = gemini_thinking_config(config.reasoning_effort, &config.model);
    let gc = GenerationConfig {
        temperature: config.temperature,
        max_output_tokens: config.max_tokens,
        response_mime_type,
        response_schema,
        thinking_config,
    };
    let generation_config = if gc.temperature.is_none()
        && gc.max_output_tokens.is_none()
        && gc.response_mime_type.is_none()
        && gc.response_schema.is_none()
        && gc.thinking_config.is_none()
    {
        None
    } else {
        Some(gc)
    };

    Request {
        contents,
        system_instruction,
        tools: gemini_tools,
        tool_config,
        generation_config,
    }
}

/// Translate the cross-provider effort into Gemini's `thinkingConfig`. The
/// knob differs by model family:
///
/// - `gemini-3*` → qualitative `thinkingLevel` (`minimal`/`low`/`medium`/`high`)
/// - `gemini-2.5*` → numeric `thinkingBudget` (`0` disabled, integer tokens)
///
/// Other models fall through to `None` (the caller omits the config and the
/// model's default takes effect).
pub(crate) fn gemini_thinking_config(
    effort: Option<ReasoningEffort>,
    model: &str,
) -> Option<ThinkingConfig> {
    let effort = effort?;
    let is_v3 = model.starts_with("gemini-3");
    let is_v25 = model.starts_with("gemini-2.5");
    if !is_v3 && !is_v25 {
        return None;
    }

    let include_thoughts = !matches!(effort, ReasoningEffort::None);

    if is_v3 {
        // Gemini 3's thinkingLevel enum. `None` → minimal (Gemini 3 Pro
        // doesn't support fully disabling thinking; minimal is the floor).
        let level = match effort {
            ReasoningEffort::None | ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max => "high",
        };
        Some(ThinkingConfig {
            thinking_level: Some(level),
            thinking_budget: None,
            include_thoughts: Some(include_thoughts),
        })
    } else {
        // Gemini 2.5 numeric budget. `None` → 0 (disabled); `Max` → a large
        // value that models cap to their own ceiling.
        let budget = match effort {
            ReasoningEffort::None => 0,
            ReasoningEffort::Minimal => 512,
            ReasoningEffort::Low => 1024,
            ReasoningEffort::Medium => 4096,
            ReasoningEffort::High => 8192,
            ReasoningEffort::XHigh => 16384,
            ReasoningEffort::Max => 24576,
        };
        Some(ThinkingConfig {
            thinking_level: None,
            thinking_budget: Some(budget),
            include_thoughts: Some(include_thoughts),
        })
    }
}

fn sanitize_schema_for_gemini(v: Value) -> Value {
    match v {
        Value::Object(mut map) => {
            // type: ["string", "null"] -> 取第一个非 null 值
            if let Some(Value::Array(types)) = map.get("type") {
                let first_non_null = types
                    .iter()
                    .find(|t| t.as_str() != Some("null"))
                    .cloned()
                    .unwrap_or(Value::String("string".into()));
                map.insert("type".into(), first_non_null);
            }
            // items: true -> items: {}
            if map.get("items").map(|v| v.is_boolean()).unwrap_or(false) {
                map.insert("items".into(), Value::Object(serde_json::Map::new()));
            }
            // 递归处理 properties
            if let Some(Value::Object(props)) = map.remove("properties") {
                let new_props: serde_json::Map<String, Value> = props
                    .into_iter()
                    .map(|(k, v)| (k, sanitize_schema_for_gemini(v)))
                    .collect();
                map.insert("properties".into(), Value::Object(new_props));
            }
            // 递归处理 items（object 情况）
            if let Some(items) = map.remove("items") {
                map.insert("items".into(), sanitize_schema_for_gemini(items));
            }
            // Gemini 不支持这些标准 JSON Schema 字段
            map.remove("$defs");
            map.remove("$schema");
            map.remove("additionalProperties");
            Value::Object(map)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Content as MsgContent, Message, ToolCall};

    fn cfg(model: &str) -> AgentConfig {
        AgentConfig {
            model: model.into(),
            ..Default::default()
        }
    }

    #[test]
    fn gemini_3_uses_thinking_level() {
        let mut c = cfg("gemini-3-pro");
        c.reasoning_effort = Some(ReasoningEffort::High);
        let req = build_gemini_request(&c, &[], &[]);
        let json = serde_json::to_value(&req).unwrap();
        let tc = &json["generationConfig"]["thinkingConfig"];
        assert_eq!(tc["thinkingLevel"], "high");
        assert!(tc["thinkingBudget"].is_null());
        assert_eq!(tc["includeThoughts"], true);
    }

    #[test]
    fn gemini_2_5_uses_thinking_budget() {
        let mut c = cfg("gemini-2.5-pro");
        c.reasoning_effort = Some(ReasoningEffort::Medium);
        let req = build_gemini_request(&c, &[], &[]);
        let json = serde_json::to_value(&req).unwrap();
        let tc = &json["generationConfig"]["thinkingConfig"];
        assert!(tc["thinkingLevel"].is_null());
        assert_eq!(tc["thinkingBudget"], 4096);
    }

    #[test]
    fn unknown_model_omits_thinking_config() {
        let mut c = cfg("gemini-1.5-flash");
        c.reasoning_effort = Some(ReasoningEffort::Max);
        let req = build_gemini_request(&c, &[], &[]);
        let json = serde_json::to_value(&req).unwrap();
        // Either generationConfig is absent entirely or thinkingConfig is null.
        assert!(
            json["generationConfig"].is_null()
                || json["generationConfig"]["thinkingConfig"].is_null()
        );
    }

    #[test]
    fn gemini_3_effort_none_emits_minimal_not_omit() {
        // Gemini 3 Pro can't fully disable; we map None → minimal.
        let mut c = cfg("gemini-3-pro");
        c.reasoning_effort = Some(ReasoningEffort::None);
        let req = build_gemini_request(&c, &[], &[]);
        let json = serde_json::to_value(&req).unwrap();
        let tc = &json["generationConfig"]["thinkingConfig"];
        assert_eq!(tc["thinkingLevel"], "minimal");
        assert_eq!(tc["includeThoughts"], false);
    }

    #[test]
    fn provider_data_gemini_parts_splice_in_order() {
        // Thought-containing functionCall + text parts must survive verbatim
        // to preserve the thoughtSignature field. Gemini 3 400s otherwise.
        let pd = serde_json::json!({
            "gemini_parts": [
                {
                    "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                    "thoughtSignature": "<Sig A>"
                },
                {
                    "text": "calling weather tool",
                    "thought": true
                }
            ]
        });
        let msgs = vec![
            Message::User(vec![MsgContent::text("weather?")]),
            Message::Assistant {
                content: Some("ignored".into()),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "ignored".into(),
                    name: "ignored".into(),
                    arguments: "{}".into(),
                }],
                provider_data: Some(pd),
            },
        ];
        let req = build_gemini_request(&cfg("gemini-3-pro"), &msgs, &[]);
        let json = serde_json::to_value(&req).unwrap();
        let model_content = json["contents"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["role"] == "model")
            .unwrap();
        let parts = model_content["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["thoughtSignature"], "<Sig A>");
        assert_eq!(parts[0]["functionCall"]["name"], "get_weather");
        assert_eq!(parts[1]["thought"], true);
    }

    #[test]
    fn document_base64_emits_inline_data() {
        use crate::request::{DocumentContent, DocumentData, UserContent};
        let msgs = vec![Message::User(vec![
            UserContent::Text {
                text: "summarize".into(),
            },
            UserContent::Document(DocumentContent {
                data: DocumentData::Base64("UERGZmFrZQ==".into()),
                mime_type: "application/pdf".into(),
                filename: None,
            }),
        ])];
        let req = build_gemini_request(&cfg("gemini-3-pro"), &msgs, &[]);
        let json = serde_json::to_value(&req).unwrap();
        let parts = json["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[1]["inline_data"]["mimeType"], "application/pdf");
        assert_eq!(parts[1]["inline_data"]["data"], "UERGZmFrZQ==");
    }

    #[test]
    fn document_url_emits_file_data() {
        use crate::request::{DocumentContent, DocumentData, UserContent};
        let msgs = vec![Message::User(vec![UserContent::Document(
            DocumentContent {
                data: DocumentData::Url("https://example.com/doc.pdf".into()),
                mime_type: "application/pdf".into(),
                filename: None,
            },
        )])];
        let req = build_gemini_request(&cfg("gemini-3-pro"), &msgs, &[]);
        let json = serde_json::to_value(&req).unwrap();
        let parts = json["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["file_data"]["mimeType"], "application/pdf");
        assert_eq!(
            parts[0]["file_data"]["fileUri"],
            "https://example.com/doc.pdf"
        );
    }
}
