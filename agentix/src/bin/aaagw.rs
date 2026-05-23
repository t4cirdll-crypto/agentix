//! `aaagw` — agentix multi-protocol LLM gateway.
//!
//! HTTP proxy that simultaneously speaks Anthropic Messages, OpenAI Chat
//! Completions, and OpenAI Responses on a single port, fanning out to a
//! shared fallback chain of upstream LLM providers.
//!
//! Each `-i <upstream>` flag opens a new upstream spec; trailing
//! `--token / --model / --base-url` flags bind to the most recent `-i`.
//! Repeated `-i` defines an ordered fallback chain.
//!
//! ```text
//! aaagw -i claude-code \
//!       -i https://api.deepseek.com/chat/completions --token $DEEPSEEK_API_KEY \
//!       127.0.0.1:7878
//! ```

use std::process::ExitCode;
use std::time::Duration;

use agentix::Provider;
use agentix::server::{AnthropicServer, UpstreamSpec};

const DEFAULT_LISTEN: &str = "127.0.0.1:7878";
const DEFAULT_PRE_COMMIT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Default)]
struct Cli {
    upstreams: Vec<UpstreamDraft>,
    listen: Option<String>,
    /// Disable the OpenAI Responses session store. Forces every request to
    /// carry full conversation history; rejects `previous_response_id`.
    stateless: bool,
    /// Path to a JSON-lines usage log. One record per completed request;
    /// see `agentix::server::UsageRecord` for the schema.
    usage_log: Option<String>,
    print_help: bool,
    print_version: bool,
}

#[derive(Debug)]
struct UpstreamDraft {
    /// Either a provider shortname (e.g. `deepseek`) or a URL (e.g.
    /// `https://api.deepseek.com/chat/completions`).
    target: String,
    token: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
}

#[derive(Debug)]
enum ParseError {
    UnknownFlag(String),
    MissingValue(String),
    OrphanFlag(String),
    BadUpstream(String),
    EmptyChain,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnknownFlag(s) => write!(f, "unknown flag: {s}"),
            ParseError::MissingValue(s) => write!(f, "flag {s} requires a value"),
            ParseError::OrphanFlag(s) => {
                write!(f, "flag {s} must follow an `-i <upstream>` declaration")
            }
            ParseError::BadUpstream(s) => write!(f, "could not parse upstream: {s}"),
            ParseError::EmptyChain => write!(f, "at least one `-i <upstream>` required"),
        }
    }
}

/// Walk argv. Each `-i <value>` opens a new draft. Trailing
/// `--token | --model | --base-url` (with `=value` or next-arg form) binds to
/// the most recent draft. One optional positional argument sets the bind
/// address. `-h | --help`, `-V | --version` are recognized.
fn parse(args: impl IntoIterator<Item = String>) -> Result<Cli, ParseError> {
    let mut iter = args.into_iter();
    // Skip argv[0] when present.
    let mut peeked: Option<String> = iter.next();
    let mut cli = Cli::default();

    let mut next = move || {
        if let Some(v) = peeked.take() {
            return Some(v);
        }
        iter.next()
    };

    // First call discards argv[0]; we only do that once.
    let _ = next();

    while let Some(arg) = next() {
        if arg == "-h" || arg == "--help" {
            cli.print_help = true;
            continue;
        }
        if arg == "-V" || arg == "--version" {
            cli.print_version = true;
            continue;
        }
        if arg == "--stateless" {
            cli.stateless = true;
            continue;
        }

        if let Some(value) = strip_eq(&arg, &["--usage-log"]) {
            cli.usage_log = Some(value);
            continue;
        }
        if arg == "--usage-log" {
            cli.usage_log = Some(next().ok_or_else(|| ParseError::MissingValue(arg.clone()))?);
            continue;
        }

        if let Some(value) = strip_eq(&arg, &["-i", "--in", "--inbound"]) {
            cli.upstreams.push(new_draft(value));
            continue;
        }
        if matches!(arg.as_str(), "-i" | "--in" | "--inbound") {
            let value = next().ok_or_else(|| ParseError::MissingValue(arg.clone()))?;
            cli.upstreams.push(new_draft(value));
            continue;
        }

        // Per-upstream trailing flags.
        let bind_to_last = |cli: &mut Cli, flag: &str, value: String| {
            let last = cli
                .upstreams
                .last_mut()
                .ok_or_else(|| ParseError::OrphanFlag(flag.to_string()))?;
            match flag {
                "--token" | "-k" => last.token = Some(value),
                "--model" | "-m" => last.model = Some(value),
                "--base-url" | "-u" => last.base_url = Some(value),
                _ => return Err(ParseError::UnknownFlag(flag.to_string())),
            }
            Ok::<_, ParseError>(())
        };

        if let Some((flag, value)) = split_eq(&arg)
            && matches!(
                flag,
                "--token" | "-k" | "--model" | "-m" | "--base-url" | "-u"
            )
        {
            bind_to_last(&mut cli, flag, value)?;
            continue;
        }

        if matches!(
            arg.as_str(),
            "--token" | "-k" | "--model" | "-m" | "--base-url" | "-u"
        ) {
            let value = next().ok_or_else(|| ParseError::MissingValue(arg.clone()))?;
            bind_to_last(&mut cli, &arg, value)?;
            continue;
        }

        if arg.starts_with('-') {
            return Err(ParseError::UnknownFlag(arg));
        }
        if cli.listen.is_none() {
            cli.listen = Some(arg);
            continue;
        }

        return Err(ParseError::UnknownFlag(arg));
    }

    Ok(cli)
}

fn new_draft(target: String) -> UpstreamDraft {
    UpstreamDraft {
        target,
        token: None,
        model: None,
        base_url: None,
    }
}

fn split_eq(arg: &str) -> Option<(&str, String)> {
    let (flag, value) = arg.split_once('=')?;
    Some((flag, value.to_string()))
}

fn strip_eq(arg: &str, flags: &[&str]) -> Option<String> {
    for f in flags {
        if let Some(rest) = arg.strip_prefix(f)
            && let Some(v) = rest.strip_prefix('=')
        {
            return Some(v.to_string());
        }
    }
    None
}

/// Materialize one [`UpstreamDraft`] into a [`UpstreamSpec`]. URLs are
/// recognized and routed via [`Provider::OpenRouter`] (agentix's catch-all
/// for OAI-compat endpoints) with the URL's host portion as `base_url`.
fn finalize_upstream(draft: UpstreamDraft) -> Result<UpstreamSpec, ParseError> {
    if draft.target.starts_with("http://") || draft.target.starts_with("https://") {
        let token = resolve_token(&draft, "OPENROUTER_API_KEY");
        let base = draft
            .base_url
            .clone()
            .unwrap_or_else(|| strip_trailing_chat_completions(&draft.target));
        let mut spec = UpstreamSpec::new(Provider::OpenRouter, token);
        spec = spec.with_base_url(base);
        if let Some(m) = draft.model {
            spec = spec.with_model(m);
        }
        spec.pre_commit_timeout = DEFAULT_PRE_COMMIT_TIMEOUT;
        return Ok(spec);
    }

    let provider = match draft.target.as_str() {
        "deepseek" => Provider::DeepSeek,
        "openai" => Provider::OpenAI,
        "anthropic" => Provider::Anthropic,
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
        other => return Err(ParseError::BadUpstream(other.to_string())),
    };

    let token_env = match provider {
        Provider::DeepSeek => "DEEPSEEK_API_KEY",
        Provider::OpenAI => "OPENAI_API_KEY",
        Provider::Anthropic => "ANTHROPIC_API_KEY",
        Provider::Gemini => "GEMINI_API_KEY",
        Provider::Kimi => "KIMI_API_KEY",
        Provider::Glm => "GLM_API_KEY",
        Provider::Minimax => "MINIMAX_API_KEY",
        Provider::Mimo => "MIMO_API_KEY",
        Provider::Grok => "GROK_API_KEY",
        Provider::OpenRouter => "OPENROUTER_API_KEY",
        #[cfg(feature = "claude-code")]
        Provider::ClaudeCode => "", // no key required
        #[cfg(feature = "codex")]
        Provider::Codex => "", // no key required
    };

    let token = resolve_token(&draft, token_env);
    let mut spec = UpstreamSpec::new(provider, token);
    if let Some(b) = draft.base_url {
        spec = spec.with_base_url(b);
    }
    if let Some(m) = draft.model {
        spec = spec.with_model(m);
    }
    spec.pre_commit_timeout = DEFAULT_PRE_COMMIT_TIMEOUT;
    Ok(spec)
}

fn resolve_token(draft: &UpstreamDraft, env_var: &str) -> String {
    if let Some(t) = &draft.token {
        return t.clone();
    }
    if !env_var.is_empty()
        && let Ok(t) = std::env::var(env_var)
    {
        return t;
    }
    String::new()
}

/// `https://api.deepseek.com/chat/completions` → `https://api.deepseek.com`.
/// Otherwise returns the URL unchanged.
fn strip_trailing_chat_completions(url: &str) -> String {
    let url = url.trim_end_matches('/');
    if let Some(stripped) = url.strip_suffix("/chat/completions") {
        return stripped.to_string();
    }
    url.to_string()
}

const HELP: &str = "\
aaagw — agentix multi-protocol LLM gateway with upstream fallback chain.

USAGE:
    aaagw -i <upstream> [--token T] [--model M] [--base-url U]
          [-i <upstream2> [--token T2] ...] ...
          [ADDR]

UPSTREAMS:
    Each `-i <upstream>` opens a new upstream. Trailing per-upstream flags
    (`--token`, `--model`, `--base-url`) bind to the most recently declared
    `-i`. Repeated `-i` declarations form a fallback chain — earlier entries
    are tried first; on any error before the upstream emits its first event,
    the next entry is tried.

    <upstream> is either:
      • a provider shortname: claude-code, anthropic, deepseek, openai,
        gemini, kimi, glm, minimax, mimo, grok, openrouter
      • a URL ending in /chat/completions — routed via Provider::OpenRouter
        with the URL host as base_url

OPTIONS:
    -i, --in <UPSTREAM>          Add an upstream to the fallback chain
    -k, --token <KEY>            API key for the most recent upstream
                                  (falls back to <PROVIDER>_API_KEY env var)
    -m, --model <MODEL>          Override the client's model field upstream
    -u, --base-url <URL>         Override the upstream's base URL
        --stateless              Disable the Responses API session store —
                                  every request must carry full input each
                                  turn; previous_response_id is rejected.
                                  Required for multi-replica deployments.
        --usage-log <PATH>       Append one JSON-lines record per completed
                                  request to PATH. Fields: ts, auth_token,
                                  wire_format, model, upstream_provider,
                                  input/output/cache/reasoning tokens,
                                  duration_ms, status. For billing.
    -h, --help                   Show this help
    -V, --version                Show version

EXAMPLE:
    # Use Claude Code as primary, DeepSeek as fallback.
    aaagw \\
        -i claude-code \\
        -i https://api.deepseek.com/chat/completions --token $DEEPSEEK_API_KEY \\
        127.0.0.1:7878
";

fn print_version() {
    println!("aaagw {}", env!("CARGO_PKG_VERSION"));
}

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();

    let args: Vec<String> = std::env::args().collect();
    let cli = match parse(args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };

    if cli.print_help {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    if cli.print_version {
        print_version();
        return ExitCode::SUCCESS;
    }

    if cli.upstreams.is_empty() {
        eprintln!("{}\n\n{HELP}", ParseError::EmptyChain);
        return ExitCode::from(2);
    }

    let mut chain = Vec::with_capacity(cli.upstreams.len());
    for d in cli.upstreams {
        match finalize_upstream(d) {
            Ok(spec) => chain.push(spec),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
        }
    }

    let listen = cli.listen.unwrap_or_else(|| DEFAULT_LISTEN.to_string());

    let listener = match tokio::net::TcpListener::bind(listen.as_str()).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {listen}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let local = match listener.local_addr() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("local_addr: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Open the usage log if requested. Shared Arc across all enabled servers.
    let usage_logger = match cli.usage_log.as_deref() {
        Some(path) => {
            match agentix::server::UsageLogger::open(path, true) {
                Ok(l) => {
                    tracing::info!(path = %path, "usage log open");
                    Some(std::sync::Arc::new(l))
                }
                Err(e) => {
                    eprintln!("failed to open usage log {path}: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        None => None,
    };

    // Merge every enabled reader's router so a single bind speaks all
    // formats simultaneously.
    let mut router = axum::Router::new();
    let mut anthropic = AnthropicServer::new(chain.clone());
    if let Some(l) = usage_logger.clone() {
        anthropic = anthropic.with_usage_logger(l);
    }
    router = router.merge(anthropic.router());

    #[cfg(feature = "server-openai-chat")]
    {
        use agentix::server::OpenAIChatServer;
        let mut openai = OpenAIChatServer::new(chain.clone());
        if let Some(l) = usage_logger.clone() {
            openai = openai.with_usage_logger(l);
        }
        router = router.merge(openai.router());
    }

    #[cfg(feature = "server-openai-responses")]
    {
        use agentix::server::OpenAIResponsesServer;
        let mut resp = OpenAIResponsesServer::new(chain.clone());
        if cli.stateless {
            resp = resp.stateless();
        }
        if let Some(l) = usage_logger.clone() {
            resp = resp.with_usage_logger(l);
        }
        router = router.merge(resp.router());
    }

    let _ = chain; // chain may be otherwise unused when only one feature is on
    let _ = cli.stateless; // unused when server-openai-responses is off
    let _ = usage_logger; // unused warning silencer for the no-features case

    tracing::info!(%local, "aaagw gateway listening");

    let serve = axum::serve(listener, router).with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
    });
    if let Err(e) = serve.await {
        eprintln!("server error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(feature = "cli")]
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,agentix=info"));
    let _ = fmt().with_env_filter(filter).try_init();
}

#[cfg(not(feature = "cli"))]
fn init_tracing() {}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_args(args: &[&str]) -> Result<Cli, ParseError> {
        let v: Vec<String> = std::iter::once("aaagw")
            .chain(args.iter().copied())
            .map(String::from)
            .collect();
        parse(v)
    }

    #[test]
    fn two_upstreams_with_token_on_second() {
        let cli = parse_args(&[
            "-i",
            "claude-code",
            "-i",
            "https://api.deepseek.com/chat/completions",
            "--token",
            "sk-x",
            "127.0.0.1:7878",
        ])
        .unwrap();
        assert_eq!(cli.upstreams.len(), 2);
        assert_eq!(cli.upstreams[0].target, "claude-code");
        assert!(cli.upstreams[0].token.is_none());
        assert_eq!(
            cli.upstreams[1].target,
            "https://api.deepseek.com/chat/completions"
        );
        assert_eq!(cli.upstreams[1].token.as_deref(), Some("sk-x"));
        assert_eq!(cli.listen.as_deref(), Some("127.0.0.1:7878"));
    }

    #[test]
    fn token_eq_form() {
        let cli = parse_args(&["-i", "deepseek", "--token=sk-x"]).unwrap();
        assert_eq!(cli.upstreams[0].token.as_deref(), Some("sk-x"));
    }

    #[test]
    fn orphan_flag_rejected() {
        let err = parse_args(&["--token", "sk-x"]).unwrap_err();
        assert!(matches!(err, ParseError::OrphanFlag(_)));
    }

    #[test]
    fn url_strip_chat_completions() {
        assert_eq!(
            strip_trailing_chat_completions("https://api.deepseek.com/chat/completions"),
            "https://api.deepseek.com"
        );
        assert_eq!(
            strip_trailing_chat_completions("https://api.example.com/v1"),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn unknown_flag() {
        let err = parse_args(&["--bogus"]).unwrap_err();
        assert!(matches!(err, ParseError::UnknownFlag(_)));
    }

    #[test]
    fn positional_listen() {
        let cli = parse_args(&["-i", "deepseek", "--token", "x", "0.0.0.0:9999"]).unwrap();
        assert_eq!(cli.listen.as_deref(), Some("0.0.0.0:9999"));
        assert_eq!(cli.upstreams.len(), 1);
        assert_eq!(cli.upstreams[0].token.as_deref(), Some("x"));
    }

    #[test]
    fn listen_flag_rejected() {
        let err = parse_args(&["--listen", "0.0.0.0:9999", "-i", "deepseek"]).unwrap_err();
        assert!(matches!(err, ParseError::UnknownFlag(_)));
    }

    #[test]
    fn duplicate_positional_listen_rejected() {
        let err = parse_args(&["-i", "deepseek", "127.0.0.1:7878", "0.0.0.0:9999"]).unwrap_err();
        assert!(matches!(err, ParseError::UnknownFlag(_)));
    }

    #[test]
    fn stateless_flag_default_false() {
        let cli = parse_args(&["-i", "deepseek"]).unwrap();
        assert!(!cli.stateless);
    }

    #[test]
    fn stateless_flag_set() {
        let cli = parse_args(&["--stateless", "-i", "deepseek"]).unwrap();
        assert!(cli.stateless);
    }

    #[test]
    fn usage_log_flag_next_arg() {
        let cli = parse_args(&["-i", "deepseek", "--usage-log", "/var/log/aaagw.jsonl"]).unwrap();
        assert_eq!(cli.usage_log.as_deref(), Some("/var/log/aaagw.jsonl"));
    }

    #[test]
    fn usage_log_flag_eq_form() {
        let cli = parse_args(&["-i", "deepseek", "--usage-log=/tmp/u.jsonl"]).unwrap();
        assert_eq!(cli.usage_log.as_deref(), Some("/tmp/u.jsonl"));
    }
}
