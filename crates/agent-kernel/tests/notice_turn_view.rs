//! A wake-up turn must tell the policy it is one. The contracts that push
//! work — a goal to advance, a plan to submit — exist for a user's request,
//! and firing them on a notice turn would keep an unattended session
//! spending on a chain nobody asked for.

use std::sync::{Arc, Mutex};

use agent_kernel::channels::UiSink;
use agent_kernel::engine::{Engine, EngineIo};
use agent_kernel::permissions::PermissionGate;
use agent_kernel::tools::registry::{ToolCtx, ToolRegistry};
use agent_kernel::turn_policy::{Effect, TurnPolicy, TurnView, Verdict};
use agent_llm_core::ChatMessage;
use agent_testkit::ScriptedProvider;

/// Records what each quiet round saw, and ends the turn there.
#[derive(Default)]
struct Recorder {
    seen: Mutex<Vec<bool>>,
}

#[async_trait::async_trait]
impl TurnPolicy for Recorder {
    fn begin_turn(&self) {}
    async fn before_dispatch(&self, _tool: &str, _is_readonly: bool) -> Option<String> {
        None
    }
    async fn on_quiet_round(&self, view: &dyn TurnView) -> Verdict {
        self.seen.lock().unwrap().push(view.is_notice_turn());
        Verdict::Finish {
            before_final: Vec::<Effect>::new(),
            after_final: Vec::new(),
        }
    }
}

fn engine_with(recorder: Arc<Recorder>) -> (Engine, EngineIo, Vec<ChatMessage>) {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("ok");
    provider.script_text("ok");
    let mut engine = Engine::new(
        provider,
        Arc::new(ToolRegistry::new()),
        Arc::new(ToolCtx::new()),
        "scripted",
        0.0,
    );
    engine.set_permissions(PermissionGate::from_mode_str("bypassPermissions"));
    engine.set_policy(Box::new(RecorderPolicy(recorder)));

    let (ui_tx, _ui_rx) = UiSink::channel();
    let (_steer_tx, steer_rx) = tokio::sync::mpsc::channel(4);
    let (_notice_tx, notice_rx) = tokio::sync::mpsc::channel(4);
    let io = EngineIo {
        ui_tx,
        steer_rx,
        notice_rx,
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    (engine, io, Vec::new())
}

/// The recording policy behind an `Arc` the test keeps.
struct RecorderPolicy(Arc<Recorder>);

#[async_trait::async_trait]
impl TurnPolicy for RecorderPolicy {
    fn begin_turn(&self) {}
    async fn before_dispatch(&self, _tool: &str, _is_readonly: bool) -> Option<String> {
        None
    }
    async fn on_quiet_round(&self, view: &dyn TurnView) -> Verdict {
        self.0.on_quiet_round(view).await
    }
}

#[tokio::test]
async fn a_notice_turn_tells_the_policy_it_is_one() {
    let recorder = Arc::new(Recorder::default());
    let (mut engine, mut io, mut history) = engine_with(recorder.clone());

    engine
        .run_turn(&mut io, &mut history, "do the thing".into())
        .await
        .expect("a user turn finishes");
    engine
        .run_notice_turn(&mut io, &mut history, vec!["[job bg-1 finished]".into()])
        .await
        .expect("a notice turn finishes");

    assert_eq!(
        *recorder.seen.lock().unwrap(),
        vec![false, true],
        "the user turn is not a notice turn; the wake-up is"
    );
}
