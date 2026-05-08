//! In-memory session store for `previous_response_id` chaining.
//!
//! When a Responses API client sets `store: true` (the default), agentix
//! retains the resolved input + output items keyed by the response ID. The
//! next request's `previous_response_id` looks up the chain and prepends it
//! to the new request's input.
//!
//! Bounded LRU eviction with a TTL — this is local-process state, not durable.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::Value;

const DEFAULT_TTL: Duration = Duration::from_secs(60 * 60); // 1 hour
const DEFAULT_CAPACITY: usize = 1024;

#[derive(Debug, Clone)]
struct Entry {
    /// All items the next turn should see as history (resolved input from this
    /// turn + output items the upstream produced).
    items: Vec<Value>,
    /// Walk-forward chain so the previous_response_id walk is O(depth).
    /// `None` means this is the root of a chain.
    parent: Option<String>,
    inserted: Instant,
}

#[derive(Debug)]
pub struct SessionStore {
    inner: Mutex<Inner>,
    ttl: Duration,
    capacity: usize,
}

#[derive(Debug)]
struct Inner {
    map: HashMap<String, Entry>,
    /// Insertion order, oldest first. Keeps eviction O(1) amortized.
    order: std::collections::VecDeque<String>,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY, DEFAULT_TTL)
    }
}

impl SessionStore {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
            }),
            ttl,
            capacity,
        }
    }

    /// Persist a turn. `id` is the response ID we'll return to the client;
    /// `items` is the resolved-input + output items.
    pub fn put(&self, id: String, items: Vec<Value>, parent: Option<String>) {
        let mut g = self.inner.lock().unwrap();
        self.evict_expired(&mut g);
        if g.map.len() >= self.capacity
            && let Some(oldest) = g.order.pop_front()
        {
            g.map.remove(&oldest);
        }
        g.order.push_back(id.clone());
        g.map.insert(
            id,
            Entry {
                items,
                parent,
                inserted: Instant::now(),
            },
        );
    }

    /// Walk back the chain rooted at `id`, returning all items in
    /// chronological order (oldest first).
    pub fn resolve(&self, id: &str) -> Option<Vec<Value>> {
        let g = self.inner.lock().unwrap();
        let mut chain: Vec<&Entry> = Vec::new();
        let mut cur = Some(id.to_string());
        while let Some(k) = cur {
            let entry = g.map.get(&k)?;
            chain.push(entry);
            cur = entry.parent.clone();
        }
        // chain is leaf→root; reverse to chronological.
        let mut out = Vec::new();
        for e in chain.iter().rev() {
            out.extend(e.items.iter().cloned());
        }
        Some(out)
    }

    fn evict_expired(&self, g: &mut Inner) {
        let now = Instant::now();
        while let Some(front) = g.order.front()
            && let Some(entry) = g.map.get(front)
            && now.duration_since(entry.inserted) >= self.ttl
        {
            let key = front.clone();
            g.order.pop_front();
            g.map.remove(&key);
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn put_and_resolve_root() {
        let store = SessionStore::default();
        store.put("r1".into(), vec![json!({"type": "message"})], None);
        let items = store.resolve("r1").unwrap();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn resolve_walks_parent_chain_in_chronological_order() {
        let store = SessionStore::default();
        store.put("r1".into(), vec![json!({"id": "first"})], None);
        store.put("r2".into(), vec![json!({"id": "second"})], Some("r1".into()));
        store.put("r3".into(), vec![json!({"id": "third"})], Some("r2".into()));
        let items = store.resolve("r3").unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["id"], "first");
        assert_eq!(items[1]["id"], "second");
        assert_eq!(items[2]["id"], "third");
    }

    #[test]
    fn missing_parent_in_chain_returns_none() {
        let store = SessionStore::default();
        store.put("r2".into(), vec![json!({})], Some("missing".into()));
        // r2's parent is missing; the resolve walk fails partway.
        assert!(store.resolve("r2").is_none());
    }

    #[test]
    fn capacity_evicts_oldest() {
        let store = SessionStore::new(2, Duration::from_secs(60));
        store.put("a".into(), vec![json!({})], None);
        store.put("b".into(), vec![json!({})], None);
        store.put("c".into(), vec![json!({})], None);
        assert_eq!(store.len(), 2);
        assert!(store.resolve("a").is_none());
        assert!(store.resolve("b").is_some());
        assert!(store.resolve("c").is_some());
    }

    #[test]
    fn ttl_expiry_runs_on_put() {
        let store = SessionStore::new(1024, Duration::from_millis(50));
        store.put("a".into(), vec![json!({})], None);
        std::thread::sleep(Duration::from_millis(80));
        // Trigger evict_expired via another put.
        store.put("b".into(), vec![json!({})], None);
        assert!(store.resolve("a").is_none());
        assert!(store.resolve("b").is_some());
    }
}
