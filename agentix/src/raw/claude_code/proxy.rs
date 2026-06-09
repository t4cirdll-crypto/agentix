//! In-process intercepting (MITM) proxy that replays a recorded tool loop into
//! the `claude` CLI as a single continuous session.
//!
//! The CLI is pointed at this proxy via `HTTPS_PROXY`/`HTTP_PROXY` and trusts
//! its CA via `NODE_EXTRA_CA_CERTS`, while still targeting the real
//! `api.anthropic.com` (so its Max-OAuth credentials are sent normally). For
//! each model call the proxy consults [`ReplayState`]:
//!
//! - recorded step → answer with the recorded assistant turn (faked SSE),
//! - first call past the recorded steps → pass through to Anthropic **untouched**
//!   for the one real generation,
//! - anything after that → a no-op `end_turn` so the CLI's loop stops without
//!   another paid upstream call.
//!
//! The genuine turn's output and real cache-usage numbers are read from the
//! CLI's own stdout (it decompresses and re-emits each Anthropic SSE event as a
//! `stream_event`), so the proxy never reads, decodes, or rewrites the upstream
//! traffic — it only fakes the recorded steps and halts.
//!
//! All the CONNECT/TLS-MITM/cert-minting plumbing comes from `hudsucker`; this
//! module only supplies the replay handler and CA bootstrap.

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::Full;
use hudsucker::certificate_authority::RcgenAuthority;
use hudsucker::hyper::{Method, Request, Response, StatusCode};
use hudsucker::{Body, HttpContext, HttpHandler, Proxy, RequestOrResponse};
use tokio::task::JoinHandle;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use hudsucker::hyper::Uri;
use hudsucker::hyper::rt::{Read, ReadBufCursor, Write};
use hudsucker::hyper_util::client::legacy::Client;
use hudsucker::hyper_util::client::legacy::connect::{Connected, Connection};
use hudsucker::hyper_util::rt::{TokioExecutor, TokioIo};
use hudsucker::rustls::pki_types::ServerName;
use hudsucker::rustls::{ClientConfig, RootCertStore};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

use super::replay::{ReplayState, TurnAction};

/// Handles the CLI's intercepted HTTP traffic against the replay state.
#[derive(Clone)]
struct ReplayHandler {
    state: Arc<ReplayState>,
}

impl HttpHandler for ReplayHandler {
    async fn handle_request(
        &mut self,
        _ctx: &HttpContext,
        req: Request<Body>,
    ) -> RequestOrResponse {
        let is_messages =
            req.method() == Method::POST && req.uri().path().ends_with("/v1/messages");
        if is_messages {
            match self.state.next_action() {
                // Faked history turn, or the no-op halt: answer locally.
                TurnAction::Fake(sse) | TurnAction::Halt(sse) => {
                    return sse_response(sse).into();
                }
                // The genuine turn: pass through untouched. The CLI's own
                // stdout carries the result, so we neither read nor alter the
                // upstream traffic here (no tee, no request tampering).
                TurnAction::Passthrough => {}
            }
        }
        req.into()
    }
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
/// caller writes [`ca_pem`](Self::ca_pem) to a temp file for `NODE_EXTRA_CA_CERTS`.
/// The genuine turn is read from the CLI's own stdout, not from the proxy.
pub(crate) struct ProxyHandle {
    pub(crate) addr: SocketAddr,
    pub(crate) ca_pem: String,
    pub(crate) task: JoinHandle<()>,
}

/// Upstream connector for the real passthrough turn.
///
/// hudsucker's default client dials the upstream directly; behind a system
/// proxy (firewalled networks, or any deployment with `HTTPS_PROXY` set) the
/// real turn can't reach `api.anthropic.com`. This connector honours
/// `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` (like `mitmdump --mode upstream:`):
/// it tunnels via HTTP `CONNECT` when a proxy is set, else dials directly, and
/// terminates TLS against the real upstream itself either way.
#[derive(Clone)]
struct UpstreamConnector {
    proxy: Option<String>,
    tls: TlsConnector,
}

type TlsTcp = tokio_rustls::client::TlsStream<TcpStream>;

/// Upstream stream: a TLS stream for the real Anthropic endpoint, or a plain
/// TCP stream for the local MCP server (loopback HTTP). Wraps both so they
/// satisfy hyper_util's `Connection` (`TokioIo` only forwards it when the inner
/// type does, and a rustls `TlsStream` doesn't). Read/Write delegate inward.
///
/// The size gap between the TLS and plain variants is expected — both are live
/// connection types and boxing the hot TLS path would only add indirection.
#[allow(clippy::large_enum_variant)]
enum MaybeTls {
    Tls(TokioIo<TlsTcp>),
    Plain(TokioIo<TcpStream>),
}

impl Read for MaybeTls {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: ReadBufCursor<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            MaybeTls::Tls(s) => Pin::new(s).poll_read(cx, buf),
            MaybeTls::Plain(s) => Pin::new(s).poll_read(cx, buf),
        }
    }
}

impl Write for MaybeTls {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            MaybeTls::Tls(s) => Pin::new(s).poll_write(cx, buf),
            MaybeTls::Plain(s) => Pin::new(s).poll_write(cx, buf),
        }
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            MaybeTls::Tls(s) => Pin::new(s).poll_flush(cx),
            MaybeTls::Plain(s) => Pin::new(s).poll_flush(cx),
        }
    }
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            MaybeTls::Tls(s) => Pin::new(s).poll_shutdown(cx),
            MaybeTls::Plain(s) => Pin::new(s).poll_shutdown(cx),
        }
    }
}

impl Connection for MaybeTls {
    fn connected(&self) -> Connected {
        Connected::new()
    }
}

impl tower_service::Service<Uri> for UpstreamConnector {
    type Response = MaybeTls;
    type Error = std::io::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let proxy = self.proxy.clone();
        let tls = self.tls.clone();
        Box::pin(async move {
            let host = uri
                .host()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "no host"))?
                .to_string();
            let is_https = uri.scheme_str() != Some("http");
            let port = uri.port_u16().unwrap_or(if is_https { 443 } else { 80 });

            // The local MCP server is reached over loopback; it must never be
            // tunnelled through the upstream proxy (only the real Anthropic
            // endpoint is). Mirrors NO_PROXY=localhost.
            let is_loopback = host == "localhost" || host == "::1" || host.starts_with("127.");
            let tcp = match proxy.as_deref().filter(|_| !is_loopback) {
                Some(p) => {
                    let mut tcp = TcpStream::connect(p).await?;
                    let req =
                        format!("CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n");
                    tcp.write_all(req.as_bytes()).await?;
                    let mut head = Vec::new();
                    let mut b = [0u8; 1];
                    loop {
                        if tcp.read(&mut b).await? == 0 {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "upstream proxy closed during CONNECT",
                            ));
                        }
                        head.push(b[0]);
                        if head.ends_with(b"\r\n\r\n") {
                            break;
                        }
                        if head.len() > 16 * 1024 {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "CONNECT response too large",
                            ));
                        }
                    }
                    let line = String::from_utf8_lossy(&head);
                    let ok = line
                        .split_whitespace()
                        .nth(1)
                        .map(|c| c.starts_with('2'))
                        .unwrap_or(false);
                    if !ok {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::ConnectionRefused,
                            format!(
                                "upstream proxy CONNECT failed: {}",
                                line.lines().next().unwrap_or("")
                            ),
                        ));
                    }
                    tcp
                }
                None => TcpStream::connect((host.as_str(), port)).await?,
            };

            if is_https {
                let server_name = ServerName::try_from(host).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string())
                })?;
                let tls_stream = tls.connect(server_name, tcp).await?;
                Ok(MaybeTls::Tls(TokioIo::new(tls_stream)))
            } else {
                Ok(MaybeTls::Plain(TokioIo::new(tcp)))
            }
        })
    }
}

/// Build the upstream client for the real passthrough turn, honouring the
/// ambient `*_PROXY` env so it works behind a system proxy. TLS is terminated
/// against the real upstream regardless.
fn build_upstream_client() -> Client<UpstreamConnector, Body> {
    let proxy = [
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ]
    .iter()
    .find_map(|k| std::env::var(k).ok())
    .filter(|v| !v.trim().is_empty())
    .map(|v| {
        v.trim()
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string()
    });

    let provider = hudsucker::rustls::crypto::aws_lc_rs::default_provider();
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = ClientConfig::builder_with_provider(Arc::new(provider))
        .with_safe_default_protocol_versions()
        .expect("rustls default protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    let tls = TlsConnector::from(Arc::new(config));

    Client::builder(TokioExecutor::new())
        .http1_title_case_headers(true)
        .http1_preserve_header_case(true)
        .build(UpstreamConnector { proxy, tls })
}

/// Spawn the replay proxy on a loopback ephemeral port.
pub(crate) async fn spawn_proxy(state: Arc<ReplayState>) -> Result<ProxyHandle, String> {
    let (authority, ca_pem) = build_ca()?;

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| format!("bind replay proxy: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("replay proxy addr: {e}"))?;

    let handler = ReplayHandler { state };

    let proxy = Proxy::builder()
        .with_listener(listener)
        .with_ca(authority)
        .with_client(build_upstream_client())
        .with_http_handler(handler)
        .build()
        .map_err(|e| format!("build replay proxy: {e}"))?;

    let task = tokio::spawn(async move {
        if let Err(e) = proxy.start().await {
            tracing::warn!(error = %e, "claude-code replay proxy exited");
        }
    });

    Ok(ProxyHandle { addr, ca_pem, task })
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
        let handle = spawn_proxy(state).await.unwrap();

        // Trust only the proxy's CA — mirrors NODE_EXTRA_CA_CERTS.
        let mut roots = RootCertStore::empty();
        for cert in rustls_pemfile::certs(&mut handle.ca_pem.as_bytes()) {
            roots.add(cert.unwrap()).unwrap();
        }
        let client_cfg = ClientConfig::builder_with_provider(Arc::new(
            hudsucker::rustls::crypto::aws_lc_rs::default_provider(),
        ))
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
}
