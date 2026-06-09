//! Example 13: Codex provider through the full `agentix::agent()` loop.
//!
//! This intentionally asks Codex to call one tool per assistant turn, then
//! waits for agentix to execute the tool and resume Codex from reconstructed
//! history. It exercises the Codex app-server provider's hardest path:
//! tool-call interrupt, process teardown, opaque provider_data round-trip,
//! `thread/inject_items`, and continuation after a tool result.
//!
//! Run with:
//!   cargo run --example 13_codex_agent_multiturn --features codex
//!
//! Optional:
//!   AGENTIX_CODEX_MODEL=gpt-5.5 cargo run --example 13_codex_agent_multiturn --features codex

use agentix::{
    AgentEvent, Content, Message, ReasoningEffort, Request, ToolBundle, UserContent, agent, tool,
};
use futures::StreamExt;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct CallLog(Arc<Mutex<Vec<String>>>);

impl CallLog {
    fn push(&self, name: &str) {
        self.0.lock().expect("call log poisoned").push(name.into());
    }

    fn snapshot(&self) -> Vec<String> {
        self.0.lock().expect("call log poisoned").clone()
    }
}

struct OrderTools {
    calls: CallLog,
}

#[tool]
impl agentix::Tool for OrderTools {
    /// Fetch a customer profile.
    /// customer_id: customer identifier, for this example use C-100.
    async fn get_customer(&self, customer_id: String) -> String {
        self.calls.push("get_customer");
        format!("{customer_id}: Ada Lovelace, tier=enterprise, home_currency=USD")
    }

    /// List invoice ids for a customer.
    /// customer_id: customer identifier returned by get_customer.
    async fn list_invoices(&self, customer_id: String) -> String {
        self.calls.push("list_invoices");
        format!("{customer_id}: INV-001 paid, INV-002 open amount=1250.50 USD")
    }

    /// Fetch invoice details.
    /// invoice_id: invoice id returned by list_invoices.
    async fn get_invoice(&self, invoice_id: String) -> String {
        self.calls.push("get_invoice");
        format!("{invoice_id}: open balance 1250.50 USD, due 2026-05-30")
    }

    /// Convert money between currencies.
    /// amount: numeric amount to convert.
    /// from: source currency code.
    /// to: target currency code.
    async fn convert_currency(&self, amount: f64, from: String, to: String) -> String {
        self.calls.push("convert_currency");
        let converted = if from == "USD" && to == "EUR" {
            amount * 0.92
        } else {
            amount
        };
        format!("{amount:.2} {from} = {converted:.2} {to}")
    }

    /// Write an audit note after all lookup/calculation tools have completed.
    /// summary: concise summary of the customer, invoice, and converted amount.
    async fn write_audit_note(&self, summary: String) -> String {
        self.calls.push("write_audit_note");
        format!("AUDIT-OK: {summary}")
    }
}

fn text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|p| {
            if let Content::Text { text } = p {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::var("AGENTIX_CODEX_MODEL").unwrap_or_else(|_| "gpt-5.5".into());
    let http = reqwest::Client::new();
    let calls = CallLog::default();
    let tools = ToolBundle::default()
        + OrderTools {
            calls: calls.clone(),
        };

    let request = Request::codex()
        .model(model)
        .reasoning_effort(ReasoningEffort::Low)
        .system_prompt(
            "You are testing an agent loop. Use the provided tools exactly as \
             instructed. Call exactly one tool per assistant turn, then wait \
             for its result before calling the next tool. Do not call tools in \
             parallel. After the audit tool succeeds, produce a final answer \
             containing the marker EXACT_FINAL.",
        );

    let history = vec![Message::User(vec![UserContent::Text {
        text: "Run this five-step workflow for customer C-100: \
               1 get_customer, 2 list_invoices, 3 get_invoice for the open \
               invoice, 4 convert the open USD balance to EUR, 5 write an \
               audit note. Then give the final result with EXACT_FINAL."
            .into(),
    }])];

    let mut stream = agent(tools, http, request, history, Some(50_000));
    let mut final_text = String::new();
    let mut tool_results = 0usize;
    let mut current_batch_tool_calls = 0usize;
    let mut batch_has_result = false;
    let mut saw_parallel_batch = false;

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::Token(t) => {
                print!("{t}");
                final_text.push_str(&t);
            }
            AgentEvent::Reasoning(t) => print!("\x1b[2m{t}\x1b[0m"),
            AgentEvent::ToolCallStart(tc) => {
                if batch_has_result {
                    current_batch_tool_calls = 0;
                    batch_has_result = false;
                }
                current_batch_tool_calls += 1;
                if current_batch_tool_calls > 1 {
                    saw_parallel_batch = true;
                }
                println!("\nCALL {}({})", tc.name, tc.arguments);
            }
            AgentEvent::ToolResult {
                name, ref content, ..
            } => {
                tool_results += 1;
                batch_has_result = true;
                println!("RESULT {name}: {}", text(content));
            }
            AgentEvent::Usage(u) => eprintln!("\n[tokens: {}]", u.total_tokens),
            AgentEvent::Done(total) => {
                eprintln!("\n[total tokens: {}]", total.total_tokens);
                break;
            }
            AgentEvent::Warning(w) => eprintln!("\n[warn] {w}"),
            AgentEvent::Error(e) => return Err(e.into()),
            AgentEvent::ToolProgress { .. } | AgentEvent::ToolCallChunk(_) => {}
        }
    }

    println!();

    let observed = calls.snapshot();
    let expected = vec![
        "get_customer",
        "list_invoices",
        "get_invoice",
        "convert_currency",
        "write_audit_note",
    ];
    if observed != expected {
        return Err(
            format!("unexpected tool order: expected {expected:?}, got {observed:?}").into(),
        );
    }
    if tool_results < expected.len() {
        return Err(format!(
            "expected at least {} tool results, got {tool_results}",
            expected.len()
        )
        .into());
    }
    if saw_parallel_batch {
        return Err("Codex called multiple tools in one assistant turn; this example expects one resume boundary per tool".into());
    }
    if !final_text.contains("EXACT_FINAL") {
        return Err("final answer did not contain EXACT_FINAL marker".into());
    }

    eprintln!("[ok] Codex agent multi-turn workflow completed");
    Ok(())
}
