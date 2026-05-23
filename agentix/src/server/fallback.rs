//! Ordered upstream fallback chain.
//!
//! Rule (per the design): any error BEFORE streaming has started → try the
//! next upstream; any error AFTER streaming has started → propagate to the
//! client. Time-based, not error-type-based.

use std::time::Duration;

use futures::StreamExt;
use futures::stream::BoxStream;
use tracing::warn;

use crate::error::ApiError;
use crate::msg::LlmEvent;
use crate::request::{Provider, Request};
use crate::types::CompleteResponse;

use super::translated::Translated;

/// One upstream in the fallback chain.
#[derive(Debug, Clone)]
pub struct UpstreamSpec {
    pub provider: Provider,
    pub api_key: String,
    /// Optional override of the provider default base URL.
    pub base_url: Option<String>,
    /// Optional override of the model field. When set, overrides whatever the
    /// client requested. When `None`, the client's `model` is forwarded as-is.
    pub model: Option<String>,
    /// How long to wait for the first event before treating this upstream as
    /// failed and falling back. Default: 30 s.
    pub pre_commit_timeout: Duration,
}

impl UpstreamSpec {
    pub fn new(provider: Provider, api_key: impl Into<String>) -> Self {
        Self {
            provider,
            api_key: api_key.into(),
            base_url: None,
            model: None,
            pre_commit_timeout: Duration::from_secs(30),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Build an agentix `Request` for one upstream, given the translated inbound
/// payload.
fn build_request(spec: &UpstreamSpec, t: &Translated) -> Request {
    let mut req = Request::new(spec.provider, spec.api_key.clone());
    let model = spec
        .model
        .clone()
        .unwrap_or_else(|| t.model_from_client.clone());
    req = req.model(model).max_tokens(t.max_tokens);
    if let Some(url) = &spec.base_url {
        req = req.base_url(url.clone());
    }
    if let Some(s) = &t.system_prompt {
        req = req.system_prompt(s.clone());
    }
    if let Some(temp) = t.temperature {
        req = req.temperature(temp);
    }
    if let Some(eff) = t.reasoning_effort {
        req = req.reasoning_effort(eff);
    }
    req = req.messages(t.messages.clone());
    req = req.tools(t.tools.clone());
    if let Some(tc) = &t.tool_choice {
        req.tool_choice = Some(tc.clone());
    }
    if !t.extra_body.is_empty() {
        req = req.extra_body(t.extra_body.clone());
    }
    req
}

/// Identifies which upstream actually served a request after a successful
/// commit. Returned alongside the response so callers (usage logger, etc.)
/// can attribute work to the right provider.
#[derive(Debug, Clone)]
pub struct CommittedUpstream {
    pub index: usize,
    pub provider: Provider,
    pub model: String,
}

/// Non-streaming dispatch with fallback. Returns the first successful upstream
/// result plus which upstream produced it, or the last error if all upstreams
/// fail.
pub async fn complete_with_fallback(
    chain: &[UpstreamSpec],
    translated: &Translated,
    http: &reqwest::Client,
) -> Result<(CompleteResponse, CommittedUpstream), ApiError> {
    let mut last_err: Option<ApiError> = None;
    for (i, spec) in chain.iter().enumerate() {
        let req = build_request(spec, translated);
        let model = req.model.clone();
        match req.complete(http).await {
            Ok(r) => {
                return Ok((
                    r,
                    CommittedUpstream {
                        index: i,
                        provider: spec.provider,
                        model,
                    },
                ));
            }
            Err(e) => {
                warn!(
                    upstream_index = i,
                    provider = ?spec.provider,
                    error = %e,
                    "non-streaming upstream failed, trying next"
                );
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| ApiError::Other("no upstreams configured".into())))
}

/// Streaming dispatch with fallback. Tries each upstream in order; on the
/// first success (peeked first event is a content event, not Error), returns
/// the full stream with the peeked event re-prepended.
///
/// Errors before commit fall back. After commit, errors propagate as part of
/// the returned stream.
pub async fn stream_with_fallback(
    chain: Vec<UpstreamSpec>,
    translated: Translated,
    http: reqwest::Client,
) -> Result<(BoxStream<'static, LlmEvent>, CommittedUpstream), ApiError> {
    let mut last_err: Option<ApiError> = None;
    for (i, spec) in chain.iter().enumerate() {
        let req = build_request(spec, &translated);
        let upstream_model = req.model.clone();
        let provider_label = format!("{:?}", spec.provider);
        let timeout = spec.pre_commit_timeout;

        // Step 1: open the stream. Failure here = pre-commit error → fallback.
        let stream_res = tokio::time::timeout(timeout, req.stream(&http)).await;
        let mut stream = match stream_res {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                warn!(upstream_index = i, provider = %provider_label, error = %e, "stream open failed, trying next");
                last_err = Some(e);
                continue;
            }
            Err(_) => {
                warn!(upstream_index = i, provider = %provider_label, "stream open timed out, trying next");
                last_err = Some(ApiError::Other(format!(
                    "upstream {} timed out before first event",
                    provider_label
                )));
                continue;
            }
        };

        // Step 2: peek the first event.
        let first = match tokio::time::timeout(timeout, stream.next()).await {
            Ok(Some(LlmEvent::Error(e))) => {
                warn!(upstream_index = i, provider = %provider_label, error = %e, "first event was Error, trying next");
                last_err = Some(ApiError::Llm(e));
                continue;
            }
            Ok(Some(ev)) => ev,
            Ok(None) => {
                warn!(upstream_index = i, provider = %provider_label, "stream ended before any event, trying next");
                last_err = Some(ApiError::Other(format!(
                    "upstream {} produced no events",
                    provider_label
                )));
                continue;
            }
            Err(_) => {
                warn!(upstream_index = i, provider = %provider_label, "first event timed out, trying next");
                last_err = Some(ApiError::Other(format!(
                    "upstream {} timed out before first event",
                    provider_label
                )));
                continue;
            }
        };

        // Step 3: commit. Re-prepend the peeked event and return the chained
        // stream + which upstream we committed to.
        let head = futures::stream::iter(std::iter::once(first));
        let combined = head.chain(stream);
        return Ok((
            combined.boxed(),
            CommittedUpstream {
                index: i,
                provider: spec.provider,
                model: upstream_model,
            },
        ));
    }
    Err(last_err.unwrap_or_else(|| ApiError::Other("no upstreams configured".into())))
}
