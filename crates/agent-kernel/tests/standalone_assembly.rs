//! A second product, assembled from the kernel's public API only: no
//! workspace, sandbox, desktop, session store or coding semantics, and every
//! tool its own. If this needs anything beyond `agent-kernel` and `agent-llm`,
//! the seam has regressed.

use std::sync::{Arc, Mutex};

use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolClass, ToolCtx, ToolRegistry, ToolResult, ToolSpec};
use agent_llm_core::{ChatMessage, LlmProvider, ModelParams, StreamChunk};
use futures::FutureExt;

/// Run a command on a remote host. It opens no local process and asks for no
/// sandbox — the kernel must neither supply nor require one.
fn ssh_exec() -> ToolSpec {
    ToolSpec {
        name: "ssh_exec",
        schema: serde_json::json!({
            "type": "object",
            "properties": { "host": { "type": "string" }, "cmd": { "type": "string" } },
            "required": ["host", "cmd"],
        }),
        readonly: false,
        class: ToolClass::Process,
        network: true,
        staged: false,
        review: None,
        exec: Arc::new(|args, _ctx| {
            async move {
                let host = args.get("host").and_then(|v| v.as_str()).unwrap_or("?");
                let cmd = args.get("cmd").and_then(|v| v.as_str()).unwrap_or("?");
                // A real worker would open an SSH channel here.
                Ok(ToolResult::text(format!("{host}$ {cmd}\nup 42 days")))
            }
            .boxed()
        }),
    }
}

/// List a remote directory.
fn list_remote() -> ToolSpec {
    ToolSpec {
        name: "list_remote",
        schema: serde_json::json!({
            "type": "object",
            "properties": { "host": { "type": "string" } },
            "required": ["host"],
        }),
        readonly: true,
        class: ToolClass::Observation,
        network: true,
        staged: false,
        review: None,
        exec: Arc::new(|_args, _ctx| {
            async move { Ok(ToolResult::text("var/\netc/\nopt/")) }.boxed()
        }),
    }
}

/// Read a remote file.
fn read_remote() -> ToolSpec {
    ToolSpec {
        name: "read_remote",
        schema: serde_json::json!({
            "type": "object",
            "properties": { "host": { "type": "string" }, "path": { "type": "string" } },
            "required": ["host", "path"],
        }),
        readonly: true,
        class: ToolClass::Observation,
        network: true,
        staged: false,
        review: None,
        exec: Arc::new(|args, _ctx| {
            async move {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
                Ok(ToolResult::text(format!("contents of {path}")))
            }
            .boxed()
        }),
    }
}

/// The worker's registry. What is absent matters as much: no kernel builtin
/// appears unless this crate registers it.
fn registry() -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    reg.register(ssh_exec());
    reg.register(read_remote());
    reg.register(list_remote());
    reg
}

/// An engine for an unattended worker.
///
/// The permission policy is the product's call: nobody is there to approve a
/// card, so this worker runs its own tools without asking. The kernel's
/// default (`default` mode) would park every non-readonly call on an approval
/// nobody can give.
fn engine(provider: Arc<dyn agent_llm_core::LlmProvider>) -> Engine {
    // A bare context: no workspace, sandbox, desktop, session or plugins.
    let ctx = Arc::new(ToolCtx::new());
    let mut engine = Engine::new(provider, Arc::new(registry()), ctx, "scripted", 0.0);
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));
    engine
}

/// One tool call, then the answer. `LlmProvider` is implementable from
/// outside the llm crate, which is the other half of the contract.
struct Scripted {
    rounds: Mutex<usize>,
    /// Tool names offered to the model, per round.
    offered: Mutex<Vec<Vec<String>>>,
}

#[async_trait::async_trait]
impl LlmProvider for Scripted {
    fn id(&self) -> &'static str {
        "scripted"
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
        tools: Option<serde_json::Value>,
        _temperature: f32,
        _reasoning_effort: Option<&str>,
        _params: &ModelParams,
    ) -> anyhow::Result<agent_llm_core::BoxStream<StreamChunk>> {
        let names = tools
            .as_ref()
            .and_then(|t| t.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        self.offered.lock().unwrap().push(names);
        let round = {
            let mut n = self.rounds.lock().unwrap();
            let r = *n;
            *n += 1;
            r
        };
        let done = || StreamChunk::Done {
            prompt_tokens: Some(5),
            completion_tokens: Some(5),
            cached_tokens: None,
        };
        let chunks = if round == 0 {
            vec![
                StreamChunk::ToolCallDelta {
                    slot: 0,
                    id: Some("call_1".into()),
                    name: Some("ssh_exec".into()),
                    args_delta: serde_json::json!({ "host": "db-1", "cmd": "uptime" }).to_string(),
                },
                done(),
            ]
        } else {
            vec![
                StreamChunk::ContentDelta("db-1 has been up 42 days.".into()),
                done(),
            ]
        };
        Ok(Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))))
    }
}

#[test]
fn the_kernel_forces_no_builtins_on_a_product() {
    let reg = registry();
    assert!(reg.spec("bash").is_none());
    assert!(reg.spec("delegate").is_none());
    assert!(reg.spec("fuzzy_patch").is_none());
    assert!(reg.spec("ssh_exec").is_some());
}

#[tokio::test]
async fn a_second_agent_runs_a_turn_with_its_own_tools() {
    let provider = Arc::new(Scripted {
        rounds: Mutex::new(0),
        offered: Mutex::new(Vec::new()),
    });
    let mut engine = engine(provider.clone());

    let (ui_tx, _ui_rx) = agent_kernel::channels::UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let mut io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx: tokio::sync::mpsc::channel(4).1,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let mut history: Vec<ChatMessage> = Vec::new();

    // Bounded: a regression that parks the turn (an approval nobody can
    // give, say) must fail the test, not hang the suite.
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        engine.run_turn(&mut io, &mut history, "check db-1".into()),
    )
    .await
    .expect("the turn must not park")
    .expect("the turn should finish");

    assert!(
        outcome.text.contains("42 days"),
        "the answer should follow the tool result: {}",
        outcome.text
    );
    assert_eq!(outcome.tool_calls_run, 1, "one tool call was dispatched");

    // What the model was offered is the product's set, not the kernel's.
    let offered = provider.offered.lock().unwrap();
    let first = offered.first().expect("the model was called");
    for own in ["ssh_exec", "read_remote", "list_remote"] {
        assert!(
            first.iter().any(|n| n == own),
            "{own} missing from {first:?}"
        );
    }
    for builtin in ["bash", "delegate", "fuzzy_patch", "test_runner"] {
        assert!(
            !first.iter().any(|n| n == builtin),
            "{builtin} leaked into {first:?}"
        );
    }
}
