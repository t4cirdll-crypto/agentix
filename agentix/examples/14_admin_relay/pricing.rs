//! OpenRouter-backed price book for the admin dashboard.
//!
//! Each route's fallback entry carries an optional `pricing_model` — a stable
//! OpenRouter model id (e.g. `anthropic/claude-opus-4`) used ONLY for cost
//! attribution, decoupled from the `target`/`model` that actually served the
//! request. We fetch OpenRouter's public model catalog (no API key needed),
//! keep per-token USD rates in memory, and cache them to disk so a restart
//! doesn't require network. Fetch failure is non-fatal: we fall back to the
//! cached copy, or to an empty book that simply prices everything at $0.
//!
//! Costing deliberately includes cache tokens. With the claude-code upstream
//! the bulk of input volume lands in `cache_creation` / `cache_read`, not in
//! `input_tokens` (which is ~constant and tiny), so pricing `input_tokens`
//! alone would under-count by orders of magnitude.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// Refetch the catalog when the cached copy is older than this.
const MAX_AGE: Duration = Duration::from_secs(24 * 3600);

/// Per-token USD rates for one model. OpenRouter quotes USD per single token.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ModelRate {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub reasoning: f64,
}

impl ModelRate {
    pub fn cost(
        &self,
        input: u64,
        output: u64,
        cache_read: u64,
        cache_write: u64,
        reasoning: u64,
    ) -> Cost {
        let input_usd = self.input * input as f64;
        let output_usd = self.output * output as f64;
        let cache_read_usd = self.cache_read * cache_read as f64;
        let cache_write_usd = self.cache_write * cache_write as f64;
        let reasoning_usd = self.reasoning * reasoning as f64;
        Cost {
            input_usd,
            output_usd,
            cache_read_usd,
            cache_write_usd,
            reasoning_usd,
            total_usd: input_usd + output_usd + cache_read_usd + cache_write_usd + reasoning_usd,
        }
    }
}

/// Cost of one request, broken out by token class.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Cost {
    pub input_usd: f64,
    pub output_usd: f64,
    pub cache_read_usd: f64,
    pub cache_write_usd: f64,
    pub reasoning_usd: f64,
    pub total_usd: f64,
}

// ── OpenRouter wire types ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    id: String,
    #[serde(default)]
    pricing: OpenRouterPricing,
}

/// OpenRouter prices are strings of USD-per-token. Fields are `Option` so a
/// missing OR explicitly-null field deserializes cleanly (rather than failing
/// the whole catalog parse) and is treated as free.
#[derive(Debug, Default, Deserialize)]
struct OpenRouterPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
    #[serde(default)]
    internal_reasoning: Option<String>,
}

impl OpenRouterPricing {
    fn to_rate(&self) -> ModelRate {
        let p = |s: &Option<String>| s.as_deref().unwrap_or("").parse::<f64>().unwrap_or(0.0);
        ModelRate {
            input: p(&self.prompt),
            output: p(&self.completion),
            cache_read: p(&self.input_cache_read),
            cache_write: p(&self.input_cache_write),
            reasoning: p(&self.internal_reasoning),
        }
    }
}

// ── On-disk cache ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CatalogOnDisk {
    /// Unix seconds when the catalog was fetched.
    fetched_at: u64,
    /// OpenRouter model id → per-token rates.
    models: BTreeMap<String, ModelRate>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Shared, refreshable price book. Cheap to `Clone` (just an `Arc`).
#[derive(Clone)]
pub struct PricingHandle {
    inner: Arc<RwLock<CatalogOnDisk>>,
    cache_path: PathBuf,
}

impl PricingHandle {
    /// Load from the on-disk cache; if missing or stale, fetch from OpenRouter.
    pub async fn load(cache_path: impl Into<PathBuf>) -> Self {
        let cache_path = cache_path.into();
        let disk = read_cache(&cache_path);
        let stale = now_secs().saturating_sub(disk.fetched_at) > MAX_AGE.as_secs();
        let empty = disk.models.is_empty();
        let handle = Self {
            inner: Arc::new(RwLock::new(disk.clone())),
            cache_path,
        };
        if empty || stale {
            handle.refresh().await;
        } else {
            info!(models = disk.models.len(), "pricing catalog loaded from cache");
        }
        handle
    }

    /// Fetch the catalog from OpenRouter and persist it. Best-effort: on
    /// failure the in-memory book is left untouched.
    pub async fn refresh(&self) {
        info!("pricing catalog stale, refreshing");
        match fetch_catalog().await {
            Ok(models) => {
                let disk = CatalogOnDisk {
                    fetched_at: now_secs(),
                    models,
                };
                if let Err(e) = write_cache(&self.cache_path, &disk) {
                    warn!(error = %e, "failed to write pricing cache");
                }
                let n = disk.models.len();
                *self.inner.write().unwrap() = disk;
                info!(models = n, "pricing catalog fetched from openrouter");
            }
            Err(e) => {
                warn!(error = %e, "openrouter fetch failed; using cached catalog");
            }
        }
    }

    /// Spawn a background task that refreshes the catalog every [`MAX_AGE`].
    pub fn spawn_periodic_refresh(&self) {
        let me = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(MAX_AGE).await;
                me.refresh().await;
            }
        });
    }

    pub fn rate(&self, pricing_model: &str) -> Option<ModelRate> {
        self.inner.read().unwrap().models.get(pricing_model).copied()
    }

    /// Cost for one request priced against `pricing_model`. Unknown model or
    /// empty book yields a zero `Cost` rather than an error.
    pub fn cost(
        &self,
        pricing_model: &str,
        input: u64,
        output: u64,
        cache_read: u64,
        cache_write: u64,
        reasoning: u64,
    ) -> Cost {
        match self.rate(pricing_model) {
            Some(r) => r.cost(input, output, cache_read, cache_write, reasoning),
            None => Cost::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().models.len()
    }
}

/// Build the per-record pricer the aggregator wants: map each record's
/// `upstream_model` to its configured `pricing_model` (via the live routes),
/// then to a USD cost against the price book. Records with no attributable
/// model, or models absent from the book, cost $0.
pub fn record_pricer(
    pricing: PricingHandle,
    routes: crate::routes::RoutesHandle,
) -> impl Fn(&crate::aggregate::LoggedRecord) -> f64 {
    move |r| {
        let Some(model) = r.upstream_model.as_deref() else {
            return 0.0;
        };
        match routes.pricing_model_for(model) {
            Some(pm) => pricing
                .cost(
                    &pm,
                    r.input_tokens,
                    r.output_tokens,
                    r.cache_read_tokens,
                    r.cache_creation_tokens,
                    r.reasoning_tokens,
                )
                .total_usd,
            None => 0.0,
        }
    }
}

async fn fetch_catalog() -> Result<BTreeMap<String, ModelRate>, String> {
    let body = reqwest::Client::new()
        .get(OPENROUTER_MODELS_URL)
        .header("user-agent", "agentix-admin-relay")
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let parsed: OpenRouterModelsResponse =
        serde_json::from_str(&body).map_err(|e| e.to_string())?;
    Ok(parsed
        .data
        .into_iter()
        .map(|m| (m.id, m.pricing.to_rate()))
        .collect())
}

fn read_cache(path: &Path) -> CatalogOnDisk {
    match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => CatalogOnDisk::default(),
    }
}

fn write_cache(path: &Path, disk: &CatalogOnDisk) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(disk).map_err(|e| e.to_string())?;
    // Reuse the routes module's crash-safe tmp-file + rename writer.
    crate::routes::atomic_write_pub(path, &body)
}
