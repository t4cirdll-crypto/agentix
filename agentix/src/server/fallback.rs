//! Ordered upstream fallback chain.
//!
//! Rule (per the design): any error BEFORE streaming has started → try the
//! next upstream; any error AFTER streaming has started → propagate to the
//! client. Time-based, not error-type-based.
//!
//! ## Chain resolution
//!
//! The server modules don't dispatch directly off a `Vec<UpstreamSpec>` —
//! they hold an `Arc<dyn ChainResolver>` and invoke `resolve(model)` at
//! request time. The default `Vec<UpstreamSpec>` resolver filters by each
//! spec's `match_patterns`. For grouped routing (one fallback chain per
//! pattern, no cross-group fallthrough), provide a custom `ChainResolver`
//! implementation that returns a different chain per `model`. See the
//! `14_admin_relay` example for one such impl with live mutation.

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
    /// Glob-style patterns this upstream serves. Empty means "catch-all".
    /// Patterns support `*` as a wildcard (e.g. `claude-*`, `*sonnet*`, `*`).
    pub match_patterns: Vec<String>,
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
            match_patterns: Vec::new(),
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

    pub fn with_match(mut self, pattern: impl Into<String>) -> Self {
        self.match_patterns.push(pattern.into());
        self
    }

    /// True if this upstream accepts a request whose `model` field is `model`.
    /// A spec with no patterns matches everything (catch-all). A spec with
    /// patterns matches if at least one pattern accepts.
    pub fn accepts_model(&self, model: &str) -> bool {
        if self.match_patterns.is_empty() {
            return true;
        }
        self.match_patterns.iter().any(|p| glob_match(p, model))
    }
}

/// Strategy for choosing which upstreams to try for a given request.
/// Servers call `resolve(model)` per request and walk the returned list
/// in order, falling back on each pre-commit error. An empty return is
/// treated as "no upstream available" → 5xx error to the client.
pub trait ChainResolver: Send + Sync + 'static {
    fn resolve(&self, model: &str) -> Vec<UpstreamSpec>;

    /// Best-effort enumeration of every upstream the resolver knows about.
    /// Used by `/v1/models` for discovery. Custom resolvers (e.g. grouped
    /// routing) should override; the default works for static chains.
    fn list_all(&self) -> Vec<UpstreamSpec> {
        self.resolve("*")
    }
}

/// Static-chain resolver: filters by each `UpstreamSpec.accepts_model`.
/// Preserves the previous (pre-grouped-routing) behaviour.
impl ChainResolver for Vec<UpstreamSpec> {
    fn resolve(&self, model: &str) -> Vec<UpstreamSpec> {
        self.iter()
            .filter(|s| s.accepts_model(model))
            .cloned()
            .collect()
    }
    fn list_all(&self) -> Vec<UpstreamSpec> {
        self.clone()
    }
}

/// Minimal `*`-wildcard matcher. Splits the pattern on `*`; each fragment
/// must appear in the input in order. Anchored at both ends unless the
/// pattern starts/ends with `*`.
pub fn glob_match(pattern: &str, input: &str) -> bool {
    if pattern == "*" || pattern.is_empty() {
        return true;
    }
    let starts_wild = pattern.starts_with('*');
    let ends_wild = pattern.ends_with('*');
    let fragments: Vec<&str> = pattern.split('*').filter(|f| !f.is_empty()).collect();
    if fragments.is_empty() {
        return true;
    }
    let mut cursor = 0;
    for (i, frag) in fragments.iter().enumerate() {
        if i == 0 && !starts_wild {
            if !input[cursor..].starts_with(frag) {
                return false;
            }
            cursor += frag.len();
            continue;
        }
        match input[cursor..].find(frag) {
            Some(pos) => cursor += pos + frag.len(),
            None => return false,
        }
    }
    if !ends_wild && cursor != input.len() {
        return false;
    }
    true
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
///
/// `chain` is the candidate list — usually produced by a `ChainResolver` for
/// the request's model. An empty `chain` returns `ApiError::Other("no
/// upstream available for model ...")`.
pub async fn complete_with_fallback(
    chain: &[UpstreamSpec],
    translated: &Translated,
    http: &reqwest::Client,
) -> Result<(CompleteResponse, CommittedUpstream), ApiError> {
    let model = translated.model_from_client.as_str();
    if chain.is_empty() {
        return Err(ApiError::Other(format!(
            "no upstream available for model {model}"
        )));
    }
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
    let model = translated.model_from_client.as_str();
    if chain.is_empty() {
        return Err(ApiError::Other(format!(
            "no upstream available for model {model}"
        )));
    }
    let candidates: Vec<(usize, UpstreamSpec)> = chain.into_iter().enumerate().collect();
    let mut last_err: Option<ApiError> = None;
    for (i, spec) in &candidates {
        let i = *i;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_star_all() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("", "anything"));
    }

    #[test]
    fn glob_prefix() {
        assert!(glob_match("claude-*", "claude-sonnet-4-5"));
        assert!(glob_match("claude-*", "claude-"));
        assert!(!glob_match("claude-*", "gpt-4o"));
    }

    #[test]
    fn glob_suffix() {
        assert!(glob_match("*-turbo", "gpt-3.5-turbo"));
        assert!(!glob_match("*-turbo", "gpt-3.5-turbo-preview"));
    }

    #[test]
    fn glob_substring() {
        assert!(glob_match("*sonnet*", "claude-sonnet-4-5"));
        assert!(!glob_match("*sonnet*", "claude-haiku-4-5"));
    }

    #[test]
    fn glob_exact() {
        assert!(glob_match("gpt-4o", "gpt-4o"));
        assert!(!glob_match("gpt-4o", "gpt-4o-mini"));
    }

    #[test]
    fn glob_multi_wildcard() {
        assert!(glob_match("claude-*-sonnet-*", "claude-3-5-sonnet-20240620"));
        assert!(!glob_match("claude-*-opus-*", "claude-3-5-sonnet-20240620"));
    }

    #[test]
    fn upstream_accepts_catchall() {
        let s = UpstreamSpec::new(Provider::DeepSeek, "");
        assert!(s.accepts_model("anything"));
    }

    #[test]
    fn upstream_accepts_specific() {
        let s = UpstreamSpec::new(Provider::Anthropic, "")
            .with_match("claude-*");
        assert!(s.accepts_model("claude-sonnet-4-5"));
        assert!(!s.accepts_model("gpt-4o"));
    }

    #[test]
    fn upstream_multiple_patterns_or() {
        let s = UpstreamSpec::new(Provider::OpenRouter, "")
            .with_match("claude-*")
            .with_match("gpt-*");
        assert!(s.accepts_model("claude-sonnet-4-5"));
        assert!(s.accepts_model("gpt-4o"));
        assert!(!s.accepts_model("deepseek-chat"));
    }
}
