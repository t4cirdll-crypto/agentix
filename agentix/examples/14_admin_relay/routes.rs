//! Grouped routing: each `[[route]]` in `aaagw.toml` declares a match
//! pattern + its OWN fallback chain. Requests look up the first route whose
//! pattern matches the request model and walk that route's fallback list.
//! Catch-all is just a `match = "*"` entry placed last.
//!
//! `RoutesHandle` is `Arc<RwLock<Routes>>` so the admin dashboard can
//! mutate the routes at runtime; changes are atomically applied + persisted
//! back to disk. Implements `ChainResolver` so it plugs directly into the
//! agentix server modules.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use agentix::Provider;
use agentix::server::UpstreamSpec;
use agentix::server::fallback::{ChainResolver, glob_match};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    /// Glob pattern matched against the inbound `model` field.
    /// Examples: `claude-*`, `*sonnet*`, `*` (catch-all).
    #[serde(rename = "match")]
    pub match_pattern: String,
    /// Ordered upstreams to try for this match. Fallback only happens
    /// WITHIN this list — never crosses into other routes.
    pub fallback: Vec<UpstreamEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamEntry {
    /// Provider shortname (`claude-code`, `deepseek`, ...) or a URL
    /// (routed through `Provider::OpenRouter`).
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Literal token. Overrides `token_env` when both are set. Avoid
    /// committing this to disk for shared secrets — prefer `token_env`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Environment variable that holds the token. Resolved at chain-build
    /// time, not stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

impl UpstreamEntry {
    fn to_spec(&self) -> Result<UpstreamSpec, String> {
        let token = self
            .token
            .clone()
            .or_else(|| self.token_env.as_deref().and_then(|e| std::env::var(e).ok()))
            .unwrap_or_default();

        if self.target.starts_with("http://") || self.target.starts_with("https://") {
            let base = self
                .base_url
                .clone()
                .unwrap_or_else(|| strip_chat_completions(&self.target));
            let mut spec = UpstreamSpec::new(Provider::OpenRouter, token);
            spec = spec.with_base_url(base);
            if let Some(m) = &self.model {
                spec = spec.with_model(m.clone());
            }
            spec.pre_commit_timeout = Duration::from_secs(30);
            return Ok(spec);
        }

        let provider = match self.target.as_str() {
            "anthropic" => Provider::Anthropic,
            "deepseek" => Provider::DeepSeek,
            "openai" => Provider::OpenAI,
            "gemini" => Provider::Gemini,
            "kimi" => Provider::Kimi,
            "glm" => Provider::Glm,
            "minimax" => Provider::Minimax,
            "mimo" => Provider::Mimo,
            "grok" => Provider::Grok,
            "openrouter" => Provider::OpenRouter,
            #[cfg(feature = "claude-code")]
            "claude-code" | "claudecode" => Provider::ClaudeCode,
            #[cfg(feature = "codex")]
            "codex" => Provider::Codex,
            other => return Err(format!("unknown upstream target: {other}")),
        };

        let mut spec = UpstreamSpec::new(provider, token);
        if let Some(b) = &self.base_url {
            spec = spec.with_base_url(b.clone());
        }
        if let Some(m) = &self.model {
            spec = spec.with_model(m.clone());
        }
        spec.pre_commit_timeout = Duration::from_secs(30);
        Ok(spec)
    }
}

fn strip_chat_completions(url: &str) -> String {
    let url = url.trim_end_matches('/');
    if let Some(s) = url.strip_suffix("/chat/completions") {
        return s.to_string();
    }
    url.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutesFile {
    #[serde(default)]
    pub route: Vec<Route>,
}

/// Shared, mutable handle to the routing rules. Cheap to `Clone` (just an
/// `Arc`). All mutations go through here so persistence stays consistent
/// with in-memory state.
#[derive(Debug, Clone)]
pub struct RoutesHandle {
    inner: Arc<RwLock<Vec<Route>>>,
    path: PathBuf,
}

impl RoutesHandle {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let body = std::fs::read_to_string(&path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let parsed: RoutesFile =
            toml::from_str(&body).map_err(|e| format!("parse {}: {e}", path.display()))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(parsed.route)),
            path,
        })
    }

    pub fn snapshot(&self) -> Vec<Route> {
        self.inner.read().unwrap().clone()
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    /// Replace the entire route list, persist to disk, return old routes
    /// for caller. Atomic write: tmp file + fsync + rename.
    pub fn replace_all(&self, new_routes: Vec<Route>) -> Result<Vec<Route>, String> {
        let body = toml::to_string_pretty(&RoutesFile { route: new_routes.clone() })
            .map_err(|e| format!("serialize routes: {e}"))?;
        atomic_write(&self.path, body.as_bytes())?;
        let mut g = self.inner.write().unwrap();
        let old = std::mem::replace(&mut *g, new_routes);
        Ok(old)
    }

    /// Add one route at the end. Returns its new index.
    pub fn append(&self, route: Route) -> Result<usize, String> {
        let mut all = self.snapshot();
        let idx = all.len();
        all.push(route);
        self.replace_all(all)?;
        Ok(idx)
    }

    /// Replace one route in place.
    pub fn update(&self, index: usize, route: Route) -> Result<(), String> {
        let mut all = self.snapshot();
        if index >= all.len() {
            return Err(format!("route index {index} out of range"));
        }
        all[index] = route;
        self.replace_all(all)?;
        Ok(())
    }

    /// Remove one route by index.
    pub fn remove(&self, index: usize) -> Result<Route, String> {
        let mut all = self.snapshot();
        if index >= all.len() {
            return Err(format!("route index {index} out of range"));
        }
        let removed = all.remove(index);
        self.replace_all(all)?;
        Ok(removed)
    }
}

impl ChainResolver for RoutesHandle {
    fn resolve(&self, model: &str) -> Vec<UpstreamSpec> {
        let routes = self.inner.read().unwrap();
        for route in routes.iter() {
            if glob_match(&route.match_pattern, model) {
                return route
                    .fallback
                    .iter()
                    .filter_map(|u| u.to_spec().ok())
                    .collect();
            }
        }
        Vec::new()
    }

    fn list_all(&self) -> Vec<UpstreamSpec> {
        let routes = self.inner.read().unwrap();
        routes
            .iter()
            .flat_map(|r| r.fallback.iter().filter_map(|u| u.to_spec().ok()))
            .collect()
    }
}

/// Write `body` to `path` atomically: write to `path.tmp`, fsync, rename.
/// Survives crashes mid-write without leaving truncated config files.
pub fn atomic_write_pub(path: &Path, body: &[u8]) -> Result<(), String> {
    atomic_write(path, body)
}

fn atomic_write(path: &Path, body: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let tmp = path.with_extension(format!(
        "tmp.{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| format!("create {}: {e}", tmp.display()))?;
        f.write_all(body)
            .map_err(|e| format!("write {}: {e}", tmp.display()))?;
        f.sync_all()
            .map_err(|e| format!("fsync {}: {e}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("rename {} → {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}
