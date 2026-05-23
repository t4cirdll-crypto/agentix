//! Read `--usage-log` (JSON lines) and roll up to the views the dashboard
//! cares about. Linear scan in memory — fine for hundreds of MB. When the
//! log gets bigger or queries get slow, swap this for sqlite ingestion.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggedRecord {
    pub ts: String,
    #[serde(default)]
    pub auth_token: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    pub wire_format: String,
    pub model: String,
    #[serde(default)]
    pub upstream_provider: Option<String>,
    #[serde(default)]
    pub upstream_model: Option<String>,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_creation_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
    #[serde(default)]
    pub duration_ms: u64,
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub streaming: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Totals {
    pub requests: u64,
    pub errors: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Totals {
    pub fn add(&mut self, r: &LoggedRecord) {
        self.requests += 1;
        if r.status != "ok" {
            self.errors += 1;
        }
        self.input_tokens += r.input_tokens;
        self.output_tokens += r.output_tokens;
        self.cache_read_tokens += r.cache_read_tokens;
        self.cache_creation_tokens += r.cache_creation_tokens;
        self.reasoning_tokens += r.reasoning_tokens;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct UserSummary {
    pub user: String,
    pub last_seen: String,
    #[serde(flatten)]
    pub totals: Totals,
}

#[derive(Debug, Clone, Serialize)]
pub struct DayPoint {
    /// `YYYY-MM-DD` (UTC).
    pub date: String,
    #[serde(flatten)]
    pub totals: Totals,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelBucket {
    /// `"<provider>/<model>"`, e.g. `"ClaudeCode/sonnet"`. `"unattributed"`
    /// when the request never committed to an upstream.
    pub key: String,
    #[serde(flatten)]
    pub totals: Totals,
}

#[derive(Debug, Clone, Serialize)]
pub struct DashboardData {
    pub overall: Totals,
    pub per_user: Vec<UserSummary>,
    pub per_day: Vec<DayPoint>,
    pub per_model: Vec<ModelBucket>,
    pub recent: Vec<LoggedRecord>,
}

/// Read the entire log into memory and roll up. `recent_n` controls how many
/// of the most-recent records to keep in `recent` (rest are aggregated only).
pub fn aggregate(path: impl AsRef<Path>, recent_n: usize) -> std::io::Result<DashboardData> {
    let body = std::fs::read_to_string(path.as_ref()).unwrap_or_default();

    let mut overall = Totals::default();
    let mut by_user: BTreeMap<String, (Totals, String)> = BTreeMap::new();
    let mut by_day: BTreeMap<String, Totals> = BTreeMap::new();
    let mut by_model: BTreeMap<String, Totals> = BTreeMap::new();
    let mut all: Vec<LoggedRecord> = Vec::new();

    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: LoggedRecord = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(_) => continue, // skip malformed lines
        };

        overall.add(&rec);

        // Per-user bucket (falls back to auth_token, or "anonymous").
        let user_key = rec
            .user
            .clone()
            .or_else(|| rec.auth_token.clone())
            .unwrap_or_else(|| "anonymous".to_string());
        let entry = by_user.entry(user_key).or_default();
        entry.0.add(&rec);
        if entry.1.is_empty() || rec.ts > entry.1 {
            entry.1 = rec.ts.clone();
        }

        // Per-day bucket — keyed on date prefix `YYYY-MM-DD`.
        let day = rec.ts.get(..10).unwrap_or("").to_string();
        if !day.is_empty() {
            by_day.entry(day).or_default().add(&rec);
        }

        // Per-model bucket.
        let model_key = match (&rec.upstream_provider, &rec.upstream_model) {
            (Some(p), Some(m)) => format!("{p}/{m}"),
            (Some(p), None) => p.clone(),
            _ => "unattributed".to_string(),
        };
        by_model.entry(model_key).or_default().add(&rec);

        all.push(rec);
    }

    // Sort the recent list newest-first, truncate.
    all.sort_by(|a, b| b.ts.cmp(&a.ts));
    all.truncate(recent_n);

    let mut per_user: Vec<UserSummary> = by_user
        .into_iter()
        .map(|(user, (totals, last_seen))| UserSummary {
            user,
            last_seen,
            totals,
        })
        .collect();
    per_user.sort_by(|a, b| {
        (b.totals.input_tokens + b.totals.output_tokens)
            .cmp(&(a.totals.input_tokens + a.totals.output_tokens))
    });

    let per_day: Vec<DayPoint> = by_day
        .into_iter()
        .map(|(date, totals)| DayPoint { date, totals })
        .collect();

    let mut per_model: Vec<ModelBucket> = by_model
        .into_iter()
        .map(|(key, totals)| ModelBucket { key, totals })
        .collect();
    per_model.sort_by(|a, b| {
        (b.totals.input_tokens + b.totals.output_tokens)
            .cmp(&(a.totals.input_tokens + a.totals.output_tokens))
    });

    Ok(DashboardData {
        overall,
        per_user,
        per_day,
        per_model,
        recent: all,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_log(records: &[serde_json::Value]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "agentix_aggregate_test_{}.jsonl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = std::fs::File::create(&p).unwrap();
        for r in records {
            writeln!(f, "{}", serde_json::to_string(r).unwrap()).unwrap();
        }
        p
    }

    #[test]
    fn aggregates_per_user_and_per_day() {
        use serde_json::json;
        let path = write_log(&[
            json!({"ts":"2026-05-22T10:00:00Z","user":"alice","auth_token":"a","wire_format":"anthropic","model":"sonnet","upstream_provider":"ClaudeCode","upstream_model":"sonnet","input_tokens":100,"output_tokens":50,"cache_read_tokens":0,"cache_creation_tokens":0,"reasoning_tokens":0,"duration_ms":1000,"status":"ok","streaming":false}),
            json!({"ts":"2026-05-22T11:00:00Z","user":"bob","auth_token":"b","wire_format":"openai_chat","model":"sonnet","upstream_provider":"ClaudeCode","upstream_model":"sonnet","input_tokens":200,"output_tokens":100,"cache_read_tokens":0,"cache_creation_tokens":0,"reasoning_tokens":0,"duration_ms":2000,"status":"ok","streaming":true}),
            json!({"ts":"2026-05-23T09:00:00Z","user":"alice","auth_token":"a","wire_format":"openai_responses","model":"sonnet","upstream_provider":"DeepSeek","upstream_model":"deepseek-chat","input_tokens":50,"output_tokens":25,"cache_read_tokens":0,"cache_creation_tokens":0,"reasoning_tokens":0,"duration_ms":500,"status":"error","error":"upstream 503","streaming":false}),
        ]);
        let data = aggregate(&path, 100).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(data.overall.requests, 3);
        assert_eq!(data.overall.errors, 1);
        assert_eq!(data.overall.input_tokens, 350);
        assert_eq!(data.overall.output_tokens, 175);

        assert_eq!(data.per_user.len(), 2);
        // Bob (300 total) > Alice (50+25 = 75 + 100+50 = 175). Wait actually
        // alice has 100+50+50+25=225, bob has 200+100=300. Bob is first.
        assert_eq!(data.per_user[0].user, "bob");
        assert_eq!(data.per_user[1].user, "alice");
        assert_eq!(data.per_user[1].totals.errors, 1);

        assert_eq!(data.per_day.len(), 2);
        assert_eq!(data.per_day[0].date, "2026-05-22");
        assert_eq!(data.per_day[1].date, "2026-05-23");

        assert!(data.per_model.iter().any(|m| m.key == "ClaudeCode/sonnet"));
        assert!(data.per_model.iter().any(|m| m.key == "DeepSeek/deepseek-chat"));

        assert_eq!(data.recent.len(), 3);
        // Newest first.
        assert_eq!(data.recent[0].ts, "2026-05-23T09:00:00Z");
    }

    #[test]
    fn falls_back_to_auth_token_when_user_missing() {
        use serde_json::json;
        let path = write_log(&[
            json!({"ts":"2026-05-22T10:00:00Z","auth_token":"sk-x","wire_format":"anthropic","model":"sonnet","upstream_provider":"ClaudeCode","upstream_model":"sonnet","input_tokens":1,"output_tokens":1,"cache_read_tokens":0,"cache_creation_tokens":0,"reasoning_tokens":0,"duration_ms":1,"status":"ok","streaming":false}),
        ]);
        let data = aggregate(&path, 100).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(data.per_user[0].user, "sk-x");
    }

    #[test]
    fn skips_malformed_lines() {
        let p = std::env::temp_dir().join("agentix_aggregate_test_bad.jsonl");
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "not json").unwrap();
        writeln!(f, "{{invalid").unwrap();
        let data = aggregate(&p, 100).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(data.overall.requests, 0);
    }
}
