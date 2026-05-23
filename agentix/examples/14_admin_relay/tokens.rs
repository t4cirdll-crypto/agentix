//! Token registry: bearer token → user attribution + access control.
//!
//! Loaded once at startup from a TOML file. Schema:
//!
//! ```toml
//! [[token]]
//! token = "sk-relay-alice"
//! user  = "alice"
//! note  = "alice's API key"           # optional
//! monthly_token_budget = 1_000_000    # optional, input + output combined
//!
//! [[token]]
//! token = "sk-relay-bob"
//! user  = "bob"
//! ```
//!
//! Any inbound request whose `Authorization: Bearer <token>` or `x-api-key`
//! header doesn't match a known token gets rejected with `401`. The matched
//! user name is forwarded into the usage log so the admin dashboard can
//! group by person. When `monthly_token_budget` is set, requests are
//! blocked with `429` once the user's current calendar-month usage reaches
//! the budget.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TokensFile {
    #[serde(default)]
    token: Vec<TokenEntryRaw>,
}

#[derive(Debug, Deserialize)]
struct TokenEntryRaw {
    token: String,
    user: String,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    monthly_token_budget: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub user: String,
    pub note: Option<String>,
    /// Combined input+output tokens per calendar month. `None` means
    /// unlimited.
    pub monthly_token_budget: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct TokenRegistry {
    map: Arc<HashMap<String, TokenEntry>>,
}

impl TokenRegistry {
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let body = std::fs::read_to_string(path.as_ref())?;
        Self::from_toml(&body)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    pub fn from_toml(body: &str) -> Result<Self, toml::de::Error> {
        let parsed: TokensFile = toml::from_str(body)?;
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
        Ok(Self { map: Arc::new(map) })
    }

    pub fn lookup(&self, token: &str) -> Option<&TokenEntry> {
        self.map.get(token)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Iterate all (token, entry) pairs. Useful for the admin dashboard's
    /// users overview.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TokenEntry)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_tokens() {
        let body = r#"
            [[token]]
            token = "sk-relay-alice"
            user = "alice"
            monthly_token_budget = 1000000

            [[token]]
            token = "sk-relay-bob"
            user = "bob"
            note = "phd student"
        "#;
        let reg = TokenRegistry::from_toml(body).unwrap();
        assert_eq!(reg.len(), 2);
        let alice = reg.lookup("sk-relay-alice").unwrap();
        assert_eq!(alice.user, "alice");
        assert_eq!(alice.monthly_token_budget, Some(1_000_000));
        let bob = reg.lookup("sk-relay-bob").unwrap();
        assert_eq!(bob.user, "bob");
        assert_eq!(bob.note.as_deref(), Some("phd student"));
        assert!(bob.monthly_token_budget.is_none());
        assert!(reg.lookup("sk-relay-unknown").is_none());
    }

    #[test]
    fn empty_file_is_empty_registry() {
        let reg = TokenRegistry::from_toml("").unwrap();
        assert!(reg.is_empty());
    }
}
