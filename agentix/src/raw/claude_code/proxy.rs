//! In-process intercepting (MITM) proxy that replays a recorded tool loop into
//! the `claude` CLI as a single continuous session.
//!
//! The CLI is pointed at this proxy via `HTTPS_PROXY`/`HTTP_PROXY` and trusts
//! its CA via `NODE_EXTRA_CA_CERTS`, while still targeting the real
//! `api.anthropic.com` (so its Max-OAuth credentials are sent normally). For
//! each model call the proxy consults [`ReplayState`]:
//!
//! - recorded step → answer with the recorded assistant turn (faked SSE),
//! - first call past the recorded steps → pass through to Anthropic for the one
//!   real generation, teeing the response bytes back to the provider so it can
//!   parse the genuine output and (crucially) the real cache-usage numbers,
//! - anything after that → a no-op `end_turn` so the CLI's loop stops without
//!   another paid upstream call.
//!
//! All the CONNECT/TLS-MITM/cert-minting plumbing comes from `hudsucker`; this
//! module only supplies the replay handler and CA bootstrap.

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use futures::TryStreamExt;
use http_body_util::{BodyExt, Full};
use hudsucker::certificate_authority::RcgenAuthority;
use hudsucker::hyper::header::{CONTENT_LENGTH, HeaderValue};
use hudsucker::hyper::{Method, Request, Response, StatusCode};
use hudsucker::{Body, HttpContext, HttpHandler, Proxy, RequestOrResponse};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::replay::{ReplayState, TurnAction};

/// Handles the CLI's intercepted HTTP traffic against the replay state.
#[derive(Clone)]
struct ReplayHandler {
    state: Arc<ReplayState>,
    /// Tee for the one real (passed-through) response body.
    tee: mpsc::UnboundedSender<Bytes>,
    /// Dynamic per-request reminder to append to the *last* message of the real
    /// request (so it never lands mid-prefix and breaks the cache).
    reminder: Option<String>,
    /// Set when the current request was passed through, so the matching
    /// response gets teed. Per request/response pair (hudsucker hands both to
    /// the same handler instance).
    armed: bool,
}

impl HttpHandler for ReplayHandler {
    async fn handle_request(&mut self, _ctx: &HttpContext, req: Request<Body>) -> RequestOrResponse {
        let is_messages =
            req.method() == Method::POST && req.uri().path().ends_with("/v1/messages");
        if is_messages {
            match self.state.next_action() {
                TurnAction::Fake(sse) | TurnAction::Halt(sse) => {
                    return sse_response(sse).into();
                }
                TurnAction::Passthrough => {
                    self.armed = true;
                    if let Some(reminder) = self.reminder.as_deref().filter(|r| !r.is_empty()) {
                        return inject_reminder(req, reminder).await;
                    }
                }
            }
        }
        req.into()
    }

    async fn handle_response(&mut self, _ctx: &HttpContext, res: Response<Body>) -> Response<Body> {
        if !self.armed {
            return res;
        }
        self.armed = false;
        let (parts, body) = res.into_parts();
        let tee = self.tee.clone();
        let stream = body
            .into_data_stream()
            .inspect_ok(move |chunk| {
                let _ = tee.send(chunk.clone());
            });
        Response::from_parts(parts, Body::from_stream(stream))
    }
}

/// Append the reminder as a trailing `text` block on the last message of the
/// real request body, placing it at the very end of the prompt (after the
/// latest tool_result) so the cached prefix is preserved. Fully guarded: any
/// parse/shape surprise forwards the original bytes unchanged so the genuine
/// generation never breaks, just loses the reminder.
async fn inject_reminder(req: Request<Body>, reminder: &str) -> RequestOrResponse {
    let (mut parts, body) = req.into_parts();
    let Ok(collected) = body.collect().await else {
        // Body already consumed/errored; nothing safe to forward.
        return Request::from_parts(parts, Body::empty()).into();
    };
    let bytes = collected.to_bytes();

    let forward = |parts: hudsucker::hyper::http::request::Parts, raw: Bytes| -> RequestOrResponse {
        Request::from_parts(parts, Body::from(Full::new(raw))).into()
    };

    let Ok(mut json) = serde_json::from_slice::<Value>(&bytes) else {
        return forward(parts, bytes);
    };
    let appended = json
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|msgs| msgs.last_mut())
        .map(|last| append_text_block(last, reminder))
        .unwrap_or(false);
    if !appended {
        return forward(parts, bytes);
    }
    let Ok(new_bytes) = serde_json::to_vec(&json) else {
        return forward(parts, bytes);
    };
    if let Ok(len) = HeaderValue::from_str(&new_bytes.len().to_string()) {
        parts.headers.insert(CONTENT_LENGTH, len);
    }
    forward(parts, Bytes::from(new_bytes))
}

/// Append a `text` content block carrying `reminder` to a message value whose
/// `content` is either a string or an array of blocks. Returns whether it was
/// applied.
fn append_text_block(message: &mut Value, reminder: &str) -> bool {
    let Some(content) = message.get_mut("content") else {
        return false;
    };
    let block = serde_json::json!({"type": "text", "text": reminder});
    match content {
        Value::Array(blocks) => blocks.push(block),
        Value::String(text) => {
            *content = Value::Array(vec![
                serde_json::json!({"type": "text", "text": text.clone()}),
                block,
            ]);
        }
        _ => return false,
    }
    true
}

fn sse_response(sse: String) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from(Full::new(Bytes::from(sse))))
        .expect("build faked sse response")
}

/// A running replay proxy. Aborting [`task`](Self::task) shuts it down; the
/// caller writes [`ca_pem`](Self::ca_pem) to a temp file for `NODE_EXTRA_CA_CERTS`
/// and drains [`real_rx`](Self::real_rx) for the genuine turn's SSE bytes.
pub(crate) struct ProxyHandle {
    pub(crate) addr: SocketAddr,
    pub(crate) ca_pem: String,
    pub(crate) real_rx: mpsc::UnboundedReceiver<Bytes>,
    pub(crate) task: JoinHandle<()>,
}

/// Spawn the replay proxy on a loopback ephemeral port.
pub(crate) async fn spawn_proxy(
    state: Arc<ReplayState>,
    reminder: Option<String>,
) -> Result<ProxyHandle, String> {
    let (authority, ca_pem) = build_ca()?;

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| format!("bind replay proxy: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("replay proxy addr: {e}"))?;

    let (tx, rx) = mpsc::unbounded_channel();
    let handler = ReplayHandler {
        state,
        tee: tx,
        reminder,
        armed: false,
    };

    let proxy = Proxy::builder()
        .with_listener(listener)
        .with_ca(authority)
        .with_rustls_client(hudsucker::rustls::crypto::aws_lc_rs::default_provider())
        .with_http_handler(handler)
        .build()
        .map_err(|e| format!("build replay proxy: {e}"))?;

    let task = tokio::spawn(async move {
        if let Err(e) = proxy.start().await {
            tracing::warn!(error = %e, "claude-code replay proxy exited");
        }
    });

    Ok(ProxyHandle {
        addr,
        ca_pem,
        real_rx: rx,
        task,
    })
}

/// Generate a throwaway CA for this run. Returns the authority (mints per-host
/// leaf certs on demand) and the CA cert PEM (for `NODE_EXTRA_CA_CERTS`).
fn build_ca() -> Result<(RcgenAuthority, String), String> {
    use hudsucker::rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, KeyPair};

    let key_pair = KeyPair::generate().map_err(|e| format!("ca keypair: {e}"))?;
    let mut params =
        CertificateParams::new(Vec::<String>::new()).map_err(|e| format!("ca params: {e}"))?;
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params
        .distinguished_name
        .push(DnType::CommonName, "agentix claude-code replay CA");
    let ca_cert = params
        .self_signed(&key_pair)
        .map_err(|e| format!("self-sign ca: {e}"))?;
    let ca_pem = ca_cert.pem();

    let authority = RcgenAuthority::new(
        key_pair,
        ca_cert,
        1_000,
        hudsucker::rustls::crypto::aws_lc_rs::default_provider(),
    );
    Ok((authority, ca_pem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::claude_code::replay::build_replay;
    use crate::request::{Content, Message, ToolCall};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_rustls::TlsConnector;
    use tokio_rustls::rustls::pki_types::ServerName;
    use tokio_rustls::rustls::{ClientConfig, RootCertStore};

    /// End-to-end exercise of the interception mechanics that can't be unit
    /// tested: a client speaks CONNECT to the proxy, completes a TLS handshake
    /// against a leaf cert minted by the proxy's throwaway CA (the
    /// `NODE_EXTRA_CA_CERTS` trust path), POSTs `/v1/messages`, and gets the
    /// recorded turn back as faked SSE — all without touching the network
    /// (recorded turns never reach the real upstream).
    #[tokio::test]
    async fn fake_turn_served_through_connect_and_tls() {
        let recorded = vec![
            Message::Assistant {
                content: None,
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    arguments: "{\"cmd\":\"ls\"}".into(),
                }],
                provider_data: None,
            },
            Message::ToolResult {
                call_id: "c1".into(),
                content: vec![Content::text("ok")],
            },
        ];
        let state = Arc::new(build_replay(&recorded, "m").unwrap());
        let handle = spawn_proxy(state, None).await.unwrap();

        // Trust only the proxy's CA — mirrors NODE_EXTRA_CA_CERTS.
        let mut roots = RootCertStore::empty();
        for cert in rustls_pemfile::certs(&mut handle.ca_pem.as_bytes()) {
            roots.add(cert.unwrap()).unwrap();
        }
        let client_cfg =
            ClientConfig::builder_with_provider(Arc::new(hudsucker::rustls::crypto::aws_lc_rs::default_provider()))
                .with_safe_default_protocol_versions()
                .unwrap()
                .with_root_certificates(roots)
                .with_no_client_auth();
        let connector = TlsConnector::from(Arc::new(client_cfg));

        // CONNECT through the proxy to the (never-contacted) upstream host.
        let mut tcp = tokio::net::TcpStream::connect(handle.addr).await.unwrap();
        tcp.write_all(
            b"CONNECT api.anthropic.com:443 HTTP/1.1\r\nHost: api.anthropic.com:443\r\n\r\n",
        )
        .await
        .unwrap();
        let mut head = [0u8; 256];
        let n = tcp.read(&mut head).await.unwrap();
        let head = String::from_utf8_lossy(&head[..n]);
        assert!(head.contains("200"), "CONNECT failed: {head}");

        // MITM TLS handshake against the proxy-minted leaf for the host.
        let server_name = ServerName::try_from("api.anthropic.com").unwrap();
        let mut tls = connector.connect(server_name, tcp).await.unwrap();

        let body = b"{\"model\":\"m\",\"messages\":[],\"stream\":true}";
        let req = format!(
            "POST /v1/messages HTTP/1.1\r\nHost: api.anthropic.com\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        tls.write_all(req.as_bytes()).await.unwrap();
        tls.write_all(body).await.unwrap();

        let mut resp = Vec::new();
        tls.read_to_end(&mut resp).await.unwrap();
        let resp = String::from_utf8_lossy(&resp);

        assert!(resp.contains("event: message_start"), "resp:\n{resp}");
        assert!(resp.contains("event: message_stop"), "resp:\n{resp}");
        // The recorded tool surfaces under the MCP namespace so the CLI routes
        // it back to our stub server.
        assert!(resp.contains("mcp__agentix__bash"), "resp:\n{resp}");

        handle.task.abort();
    }

    #[test]
    fn append_text_block_handles_array_and_string() {
        // Array content (tool_result turn): reminder appended after the blocks.
        let mut arr = serde_json::json!({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]
        });
        assert!(append_text_block(&mut arr, "REMINDER"));
        let blocks = arr["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1]["text"], "REMINDER");

        // String content: promoted to an array preserving the original text.
        let mut s = serde_json::json!({"role": "user", "content": "hello"});
        assert!(append_text_block(&mut s, "REMINDER"));
        let blocks = s["content"].as_array().unwrap();
        assert_eq!(blocks[0]["text"], "hello");
        assert_eq!(blocks[1]["text"], "REMINDER");

        // Missing content: no-op, reported as not applied.
        let mut none = serde_json::json!({"role": "user"});
        assert!(!append_text_block(&mut none, "REMINDER"));
    }
}
