//! Token registry: bearer token → user attribution + access control.
//!
//! Mutable at runtime — the admin dashboard adds/removes/updates tokens
//! and the registry persists every change back to `tokens.toml`.
//!
//! ```toml
//! [[token]]
//! token = "sk-relay-alice"
//! user  = "alice"
//! note  = "alice's API key"           # optional
//! monthly_token_budget = 1_000_000    # optional, input + output combined
//! ```
//!
//! Any inbound request whose `Authorization: Bearer <token>` or `x-api-key`
//! header doesn't match a known token gets rejected with `401`. The matched
//! user name is forwarded into the usage log so the admin dashboard can
//! group by person. When `monthly_token_budget` is set, requests are
//! blocked with `429` once the user's current calendar-month usage reaches
//! the budget.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default)]
struct TokensFile {
    #[serde(default)]
    token: Vec<TokenEntryRaw>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TokenEntryRaw {
    token: String,
    user: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    monthly_token_budget: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEntry {
    pub user: String,
    pub note: Option<String>,
    /// Combined input+output tokens per calendar month. `None` means
    /// unlimited.
    pub monthly_token_budget: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TokenRegistry {
    map: Arc<RwLock<HashMap<String, TokenEntry>>>,
    path: PathBuf,
}

impl TokenRegistry {
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let body = std::fs::read_to_string(&path).unwrap_or_default();
        let parsed: TokensFile = if body.trim().is_empty() {
            TokensFile::default()
        } else {
            toml::from_str(&body)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?
        };
        let mut map = HashMap::with_capacity(parsed.token.len());
        for t in parsed.token {
            map.insert(
                t.token,
                TokenEntry {
                    user: t.user,
                    note: t.note,
                    monthly_token_budget: t.monthly_token_budget,
                },
            );
        }
        Ok(Self {
            map: Arc::new(RwLock::new(map)),
            path,
        })
    }

    pub fn lookup(&self, token: &str) -> Option<TokenEntry> {
        self.map.read().unwrap().get(token).cloned()
    }

    pub fn len(&self) -> usize {
        self.map.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.read().unwrap().is_empty()
    }

    /// Snapshot of all `(token, entry)` pairs. Cheap for small registries.
    pub fn snapshot(&self) -> Vec<(String, TokenEntry)> {
        let g = self.map.read().unwrap();
        g.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Insert or replace a token. Persists to `tokens.toml`.
    pub fn upsert(&self, token: String, entry: TokenEntry) -> Result<(), String> {
        {
            let mut g = self.map.write().unwrap();
            g.insert(token, entry);
        }
        self.persist()
    }

    /// Remove a token. Returns `true` if it existed. Persists.
    pub fn remove(&self, token: &str) -> Result<bool, String> {
        let existed = {
            let mut g = self.map.write().unwrap();
            g.remove(token).is_some()
        };
        if existed {
            self.persist()?;
        }
        Ok(existed)
    }

    fn persist(&self) -> Result<(), String> {
        let mut entries: Vec<TokenEntryRaw> = self
            .map
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| TokenEntryRaw {
                token: k.clone(),
                user: v.user.clone(),
                note: v.note.clone(),
                monthly_token_budget: v.monthly_token_budget,
            })
            .collect();
        // Stable on-disk ordering: sort by user, then by token.
        entries.sort_by(|a, b| {
            a.user
                .cmp(&b.user)
                .then_with(|| a.token.cmp(&b.token))
        });
        let body = toml::to_string_pretty(&TokensFile { token: entries })
            .map_err(|e| format!("serialize tokens: {e}"))?;
        crate::routes::atomic_write_pub(&self.path, body.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_token_file(body: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "agentix_tokens_test_{}.toml",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn parses_two_tokens() {
        let p = temp_token_file(
            r#"
[[token]]
token = "sk-relay-alice"
user = "alice"
monthly_token_budget = 1000000

[[token]]
token = "sk-relay-bob"
user = "bob"
note = "phd student"
"#,
        );
        let reg = TokenRegistry::from_file(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert_eq!(reg.len(), 2);
        let alice = reg.lookup("sk-relay-alice").unwrap();
        assert_eq!(alice.user, "alice");
        assert_eq!(alice.monthly_token_budget, Some(1_000_000));
        let bob = reg.lookup("sk-relay-bob").unwrap();
        assert_eq!(bob.note.as_deref(), Some("phd student"));
    }

    #[test]
    fn upsert_and_remove_persists() {
        let p = temp_token_file("");
        let reg = TokenRegistry::from_file(&p).unwrap();
        reg.upsert(
            "sk-relay-new".into(),
            TokenEntry {
                user: "new".into(),
                note: None,
                monthly_token_budget: None,
            },
        )
        .unwrap();
        // Re-open from disk: change is persisted.
        let reg2 = TokenRegistry::from_file(&p).unwrap();
        assert!(reg2.lookup("sk-relay-new").is_some());

        assert!(reg2.remove("sk-relay-new").unwrap());
        let reg3 = TokenRegistry::from_file(&p).unwrap();
        assert!(reg3.lookup("sk-relay-new").is_none());

        let _ = std::fs::remove_file(&p);
    }
}
