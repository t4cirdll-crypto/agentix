//! Per-request usage logging for billing.
//!
//! Every successful or failed request through any of the three server formats
//! writes one JSON line to the configured log file. The schema is stable so
//! downstream tools (cron parsers, billing scripts) can rely on it.
//!
//! ```jsonl
//! {"ts":"2026-05-23T10:30:00Z","auth_token":"sk-relay-abc","wire_format":"anthropic","model":"claude-sonnet-4-5","upstream_provider":"ClaudeCode","upstream_model":"sonnet","input_tokens":123,"output_tokens":456,"cache_read_tokens":0,"cache_creation_tokens":0,"reasoning_tokens":0,"duration_ms":1234,"status":"ok","error":null,"streaming":true}
//! ```
//!
//! Token counts come from agentix's `UsageStats` (which is populated from the
//! upstream's `usage` field when it provides one). For upstreams that don't
//! report usage, all token fields are zero — caller can detect this and use
//! `estimate_tokens()` as a fallback if desired.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::msg::LlmEvent;
use crate::request::Provider;
use crate::types::UsageStats;

use super::fallback::CommittedUpstream;

/// One record per completed request (success or failure).
#[derive(Debug, Serialize, Clone)]
pub struct UsageRecord {
    /// ISO 8601 UTC timestamp (RFC 3339).
    pub ts: String,
    /// Bearer token from the client's `Authorization` header, with the
    /// `Bearer ` prefix stripped. `None` if the client sent no auth header.
    pub auth_token: Option<String>,
    /// User name resolved by the token registry (when token auth is
    /// enabled). `None` when the proxy runs without `--tokens`.
    pub user: Option<String>,
    /// Inbound wire format: `"anthropic" | "openai_chat" | "openai_responses"`.
    pub wire_format: &'static str,
    /// Model field as sent by the client (NOT the upstream rewrite).
    pub model: String,
    /// Which agentix provider actually served the request, formatted as e.g.
    /// `"ClaudeCode"`, `"Anthropic"`, `"DeepSeek"`. `None` if all upstreams
    /// failed before commit.
    pub upstream_provider: Option<String>,
    /// Model name sent to the upstream (after any `--model` override). `None`
    /// if upstream commit never happened.
    pub upstream_model: Option<String>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cache_read_tokens: usize,
    pub cache_creation_tokens: usize,
    pub reasoning_tokens: usize,
    pub duration_ms: u64,
    /// `"ok"` if the request reached `Done` cleanly; `"error"` if it ended
    /// with `LlmEvent::Error` or all-upstreams-failed.
    pub status: &'static str,
    /// Error message if `status == "error"`, else `None`.
    pub error: Option<String>,
    /// True if `stream: true` was set on the request.
    pub streaming: bool,
}

impl UsageRecord {
    pub fn builder(wire_format: &'static str, model: impl Into<String>) -> UsageRecordBuilder {
        UsageRecordBuilder {
            wire_format,
            model: model.into(),
            auth_token: None,
            user: None,
            upstream_provider: None,
            upstream_model: None,
            usage: UsageStats::default(),
            duration_ms: 0,
            status: "ok",
            error: None,
            streaming: false,
        }
    }
}

#[derive(Debug)]
pub struct UsageRecordBuilder {
    wire_format: &'static str,
    model: String,
    auth_token: Option<String>,
    user: Option<String>,
    upstream_provider: Option<Provider>,
    upstream_model: Option<String>,
    usage: UsageStats,
    duration_ms: u64,
    status: &'static str,
    error: Option<String>,
    streaming: bool,
}

impl UsageRecordBuilder {
    pub fn auth_token(mut self, t: Option<String>) -> Self {
        self.auth_token = t;
        self
    }
    pub fn user(mut self, u: Option<String>) -> Self {
        self.user = u;
        self
    }
    pub fn upstream(mut self, provider: Provider, model: impl Into<String>) -> Self {
        self.upstream_provider = Some(provider);
        self.upstream_model = Some(model.into());
        self
    }
    pub fn usage(mut self, u: UsageStats) -> Self {
        self.usage = u;
        self
    }
    pub fn duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }
    pub fn streaming(mut self, s: bool) -> Self {
        self.streaming = s;
        self
    }
    pub fn ok(mut self) -> Self {
        self.status = "ok";
        self.error = None;
        self
    }
    pub fn error(mut self, msg: impl Into<String>) -> Self {
        self.status = "error";
        self.error = Some(msg.into());
        self
    }
    pub fn build(self) -> UsageRecord {
        UsageRecord {
            ts: rfc3339_now(),
            auth_token: self.auth_token,
            user: self.user,
            wire_format: self.wire_format,
            model: self.model,
            upstream_provider: self.upstream_provider.map(format_provider),
            upstream_model: self.upstream_model,
            input_tokens: self.usage.prompt_tokens,
            output_tokens: self.usage.completion_tokens,
            cache_read_tokens: self.usage.cache_read_tokens,
            cache_creation_tokens: self.usage.cache_creation_tokens,
            reasoning_tokens: self.usage.reasoning_tokens,
            duration_ms: self.duration_ms,
            status: self.status,
            error: self.error,
            streaming: self.streaming,
        }
    }
}

fn format_provider(p: Provider) -> String {
    // Debug repr keeps casing consistent with how we already format it in
    // server logs and matches how the CLI's --to flag accepts values when
    // lowercased.
    format!("{:?}", p)
}

/// Append-only JSON-lines logger. Single mutex for cross-server writes.
/// fsync on every write is too slow under load; we buffer through
/// `BufWriter` and flush on drop. Set fsync via `flush_on_each_write` if
/// you need every record durable before the request returns.
#[derive(Debug)]
pub struct UsageLogger {
    inner: Mutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    writer: BufWriter<std::fs::File>,
    flush_each: bool,
}

impl UsageLogger {
    /// Open `path` in append mode (creates the file if missing). Each line
    /// is one JSON record. `flush_each` controls whether we call `flush()`
    /// after every write — true for billing-grade durability, false for max
    /// throughput.
    pub fn open(path: impl Into<PathBuf>, flush_each: bool) -> std::io::Result<Self> {
        let path = path.into();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            inner: Mutex::new(Inner {
                writer: BufWriter::new(file),
                flush_each,
            }),
        })
    }

    pub fn log(&self, record: &UsageRecord) {
        let Ok(mut g) = self.inner.lock() else {
            return;
        };
        if let Ok(line) = serde_json::to_string(record) {
            let _ = g.writer.write_all(line.as_bytes());
            let _ = g.writer.write_all(b"\n");
            if g.flush_each {
                let _ = g.writer.flush();
            }
        }
    }

    pub fn flush(&self) {
        if let Ok(mut g) = self.inner.lock() {
            let _ = g.writer.flush();
        }
    }
}

impl Drop for UsageLogger {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Extracts a Bearer token from an `Authorization` header value. Returns
/// `None` for missing header, missing `Bearer ` prefix, or empty token.
pub fn parse_bearer_token(header: Option<&str>) -> Option<String> {
    let raw = header?.trim();
    let token = raw
        .strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
        .unwrap_or(raw);
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

/// Identity attached to a request by the proxy-token auth middleware.
/// Handlers read this from request extensions to enrich usage records.
#[derive(Debug, Clone)]
pub struct AuthedUser {
    pub token: String,
    pub user: String,
}

/// Resolve a client identifier from request headers. Tries `Authorization:
/// Bearer ...` first (OpenAI convention), then `x-api-key` (Anthropic
/// convention). Returns `None` if neither is set.
pub fn extract_client_token(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(t) = parse_bearer_token(
        headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok()),
    ) {
        return Some(t);
    }
    let xapi = headers.get("x-api-key").and_then(|v| v.to_str().ok())?;
    let trimmed = xapi.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Per-request state used by the handlers to assemble a single
/// [`UsageRecord`]. Cheap to construct at request entry; `finalize` writes
/// the record exactly once.
#[derive(Debug)]
pub struct UsageTracker {
    logger: Option<Arc<UsageLogger>>,
    started_at: Instant,
    wire_format: &'static str,
    model: String,
    auth_token: Option<String>,
    user: Option<String>,
    streaming: bool,
    committed: Option<CommittedUpstream>,
    last_usage: UsageStats,
    status: &'static str,
    error: Option<String>,
}

impl UsageTracker {
    pub fn new(
        logger: Option<Arc<UsageLogger>>,
        wire_format: &'static str,
        model: impl Into<String>,
        auth_token: Option<String>,
        streaming: bool,
    ) -> Self {
        Self {
            logger,
            started_at: Instant::now(),
            wire_format,
            model: model.into(),
            auth_token,
            user: None,
            streaming,
            committed: None,
            last_usage: UsageStats::default(),
            status: "ok",
            error: None,
        }
    }

    pub fn set_user(&mut self, user: Option<String>) {
        self.user = user;
    }

    pub fn set_committed(&mut self, c: CommittedUpstream) {
        self.committed = Some(c);
    }

    pub fn set_usage(&mut self, u: UsageStats) {
        self.last_usage = u;
    }

    /// Update the tracker from one `LlmEvent`. Returns `true` if the event
    /// signals end-of-stream (Done or Error) so the caller knows when to
    /// finalize.
    pub fn observe(&mut self, ev: &LlmEvent) -> bool {
        match ev {
            LlmEvent::Usage(u) => {
                self.last_usage = u.clone();
                false
            }
            LlmEvent::Done => true,
            LlmEvent::Error(e) => {
                self.status = "error";
                self.error = Some(e.clone());
                true
            }
            _ => false,
        }
    }

    pub fn mark_error(&mut self, msg: impl Into<String>) {
        self.status = "error";
        self.error = Some(msg.into());
    }

    /// Write exactly one record. Safe to call from any thread; idempotent in
    /// the sense that the tracker should be dropped after finalize.
    pub fn finalize(self) {
        let Some(logger) = self.logger else {
            return;
        };
        let duration_ms = self.started_at.elapsed().as_millis() as u64;
        let mut builder = UsageRecord::builder(self.wire_format, self.model)
            .auth_token(self.auth_token)
            .user(self.user)
            .usage(self.last_usage)
            .duration_ms(duration_ms)
            .streaming(self.streaming);
        if let Some(c) = self.committed {
            builder = builder.upstream(c.provider, c.model);
        }
        let rec = if self.status == "error" {
            let msg = self.error.unwrap_or_else(|| "unknown error".to_string());
            builder.error(msg).build()
        } else {
            builder.ok().build()
        };
        logger.log(&rec);
    }
}

fn rfc3339_now() -> String {
    // Minimal RFC 3339 formatter; avoids pulling in `chrono` just for this.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();

    // Days since unix epoch (1970-01-01).
    let days = secs.div_euclid(86_400);
    let sec_of_day = secs.rem_euclid(86_400);
    let (h, m, s) = (
        (sec_of_day / 3600) as u32,
        ((sec_of_day / 60) % 60) as u32,
        (sec_of_day % 60) as u32,
    );

    let (year, month, day) = days_to_ymd(days);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, h, m, s, millis
    )
}

/// Convert days since 1970-01-01 to (year, month, day). Proleptic Gregorian.
fn days_to_ymd(days_since_epoch: i64) -> (i32, u32, u32) {
    // Howard Hinnant's algorithm, simplified.
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y } as i32;
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ymd_known_dates() {
        // 1970-01-01
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
        // 2000-01-01 — exactly 10957 days after epoch
        assert_eq!(days_to_ymd(10957), (2000, 1, 1));
        // 2024-02-29 (leap year)
        assert_eq!(days_to_ymd(19782), (2024, 2, 29));
    }

    #[test]
    fn rfc3339_format() {
        let s = rfc3339_now();
        // Sanity: starts with 4-digit year and has the Z suffix.
        assert!(s.len() >= 24, "got {s}");
        assert!(s.ends_with('Z'), "got {s}");
        assert_eq!(&s[4..5], "-");
    }

    #[test]
    fn logger_writes_jsonl() {
        let tmp = tempfile_path();
        let logger = UsageLogger::open(&tmp, true).unwrap();
        let rec = UsageRecord::builder("anthropic", "claude-sonnet-4-5")
            .auth_token(Some("sk-relay-test".into()))
            .upstream(Provider::Anthropic, "claude-sonnet-4-5")
            .usage(UsageStats {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
                ..Default::default()
            })
            .duration_ms(2000)
            .streaming(true)
            .ok()
            .build();
        logger.log(&rec);
        logger.flush();
        drop(logger);

        let content = std::fs::read_to_string(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed["auth_token"], "sk-relay-test");
        assert_eq!(parsed["wire_format"], "anthropic");
        assert_eq!(parsed["input_tokens"], 100);
        assert_eq!(parsed["output_tokens"], 50);
        assert_eq!(parsed["status"], "ok");
        assert_eq!(parsed["streaming"], true);
        assert_eq!(parsed["upstream_provider"], "Anthropic");
    }

    #[test]
    fn logger_records_error_status() {
        let tmp = tempfile_path();
        let logger = UsageLogger::open(&tmp, true).unwrap();
        let rec = UsageRecord::builder("openai_chat", "x").error("upstream 503").build();
        logger.log(&rec);
        drop(logger);
        let content = std::fs::read_to_string(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["status"], "error");
        assert_eq!(parsed["error"], "upstream 503");
    }

    fn tempfile_path() -> PathBuf {
        let dir = std::env::temp_dir();
        let name = format!(
            "agentix_usage_test_{}.jsonl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        dir.join(name)
    }
}
