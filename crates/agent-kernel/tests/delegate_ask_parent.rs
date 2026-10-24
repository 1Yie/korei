//! `ask_parent` / `answer_child` — a child that parks mid-task and resumes on
//! the parent's answer, end to end with a scripted provider.

use std::sync::Arc;

use agent_ipc::events::UiEvent;
use agent_kernel::channels::UiSink;
use agent_kernel::tools::delegate::{ChildRoute, SubagentSpawner};
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_llm_core::types::StreamChunk;
use agent_testkit::ScriptedProvider;

fn spawner(provider: Arc<ScriptedProvider>) -> SubagentSpawner {
    let registry = Arc::new(ToolRegistry::child_builtins());
    let route = Arc::new(std::sync::RwLock::new(ChildRoute {
        provider,
        model: "test".into(),
        temperature: 0.0,
        context_window: 256_000,
        thinking_level: None,
        thinking_level_map: None,
        model_params: Default::default(),
        model_input: Vec::new(),
        models: Default::default(),
    }));
    SubagentSpawner::new(route, registry.clone(), registry)
}

fn child_asks(provider: &ScriptedProvider, question: &str) {
    provider.push_script(vec![
        StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("call_ask".into()),
            name: Some("ask_parent".into()),
            args_delta: serde_json::json!({ "question": question }).to_string(),
        },
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        },
    ]);
}

/// The id the waiting notice tells the parent to answer.
fn child_id_of(text: &str) -> u64 {
    const NEEDLE: &str = "child_id ";
    let at = text.find(NEEDLE).expect("the notice names the child") + NEEDLE.len();
    text[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("the notice carries a numeric child id")
}

/// A child's question ends the `delegate` call with a notice, and the parent's
/// `answer_child` resumes the same child and returns its report.
#[tokio::test]
async fn a_question_parks_the_child_and_answer_child_resumes_it() {
    let provider = Arc::new(ScriptedProvider::new());
    child_asks(&provider, "which crate should I use?");
    provider.script_text("child adopted korei");
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider.clone()));
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let waiting = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (delegate.exec)(serde_json::json!({ "task": "inspect" }), ctx.clone()),
    )
    .await
    .expect("a parked child must hand its question back")
    .expect("asking is not an error");
    assert!(
        waiting.content.contains("is waiting for you"),
        "{}",
        waiting.content
    );
    assert!(
        waiting.content.contains("which crate should I use?"),
        "the question must reach the parent: {}",
        waiting.content
    );
    let child_id = child_id_of(&waiting.content);

    let answer = ToolRegistry::with_builtins()
        .spec("answer_child")
        .expect("the parent holds answer_child");
    let report = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (answer.exec)(
            serde_json::json!({ "child_id": child_id, "answer": "use korei" }),
            ctx,
        ),
    )
    .await
    .expect("the resumed child must finish")
    .expect("the child finished cleanly");
    assert!(
        report.content.contains("child adopted korei"),
        "answer_child returns the child's report: {}",
        report.content
    );

    // The child was shown the answer as its `ask_parent` result, not as prose.
    let seen = provider.seen_messages.lock().unwrap();
    let last = seen.last().expect("the child sampled twice");
    assert!(
        last.iter().any(|m| m.contains("use korei")),
        "the answer must arrive as the child's tool result: {last:?}"
    );
}

/// A cancel while the child is parked tears it down and closes its card — the
/// card was opened for the child's whole lifetime, not for one tool call.
#[tokio::test]
async fn a_cancel_while_parked_closes_the_child_card() {
    let provider = Arc::new(ScriptedProvider::new());
    child_asks(&provider, "wait on me");
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (sink, mut rx) = UiSink::channel();
    let mut ctx = ToolCtx::new();
    ctx.cancel = Some(cancel.clone());
    ctx.install_subagent(spawner(provider));
    ctx.install_ui(sink);
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let waiting = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (delegate.exec)(serde_json::json!({ "task": "wait" }), ctx),
    )
    .await
    .expect("the child must park first")
    .expect("asking is not an error");
    let child_id = child_id_of(&waiting.content);

    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    let mut opened = false;
    let mut closed = false;
    for _ in 0..100 {
        while let Some(ev) = rx.try_recv() {
            match ev {
                UiEvent::ToolCallStarted {
                    child_id: Some(id), ..
                } if id == child_id => opened = true,
                UiEvent::ToolCallFinished {
                    ok,
                    child_id: Some(id),
                    ..
                } if id == child_id => closed = closed || !ok,
                _ => {}
            }
        }
        if opened && closed {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(opened, "the child's card must have opened");
    assert!(closed, "a cancelled child must close its card as failed");
}

/// Answering a child that does not exist — or is not this session's — is an
/// error, not a silent no-op the parent would wait on forever.
#[tokio::test]
async fn answering_an_unknown_child_errors() {
    let ctx = Arc::new(ToolCtx::new());
    let answer = ToolRegistry::with_builtins()
        .spec("answer_child")
        .expect("the parent holds answer_child");
    let err = (answer.exec)(
        serde_json::json!({ "child_id": 987_654, "answer": "hello" }),
        ctx,
    )
    .await
    .expect_err("an unknown child cannot be answered");
    assert!(
        err.to_string().contains("no running subagent with id 987654"),
        "{err}"
    );
}

/// One child of a parallel batch asks while its sibling is still working: the
/// batch hands the question back at once instead of waiting on the sibling,
/// the parent can answer it meanwhile, and the sibling's report arrives later
/// through the context's child notifier.
#[tokio::test]
async fn a_question_in_a_batch_returns_without_waiting_on_siblings() {
    let provider = Arc::new(ScriptedProvider::new());
    // The gate holds whichever child samples first; the other one asks.
    let gate = Arc::new(tokio::sync::Notify::new());
    *provider.gate.lock().unwrap() = Some(gate.clone());
    child_asks(&provider, "which crate should I use?");
    provider.script_text("asker adopted korei");
    provider.script_text("sibling report");
    let (notice_tx, mut notices) = tokio::sync::mpsc::unbounded_channel::<String>();
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider.clone()));
    agent_kernel::tools::delegate::install_child_notifier(&mut ctx, move |text| {
        let _ = notice_tx.send(text);
    });
    let ctx = Arc::new(ctx);

    let delegate = agent_kernel::tools::delegate::spec();
    let batch = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (delegate.exec)(
            serde_json::json!({ "tasks": ["a", "b"], "readonly": true }),
            ctx.clone(),
        ),
    )
    .await
    .expect("a question must not wait on a sibling still running")
    .expect("asking is not an error");
    let report = &batch.content;
    assert!(report.contains("1 waiting, 1 running"), "{report}");
    assert!(report.contains("] waiting ·"), "{report}");
    assert!(report.contains("] running ──"), "{report}");
    assert!(report.contains("which crate should I use?"), "{report}");
    let waiting_at = report.find("] waiting ·").expect("the asker's section");
    let asker = child_id_of(&report[waiting_at..]);

    let answer = ToolRegistry::with_builtins()
        .spec("answer_child")
        .expect("the parent holds answer_child");
    let resumed = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (answer.exec)(
            serde_json::json!({ "child_id": asker, "answer": "use korei" }),
            ctx,
        ),
    )
    .await
    .expect("the asker resumes while its sibling is still held")
    .expect("the asker finished cleanly");
    assert!(resumed.content.contains("asker adopted korei"), "{}", resumed.content);
    assert!(notices.try_recv().is_err(), "the sibling is still held");

    gate.notify_one();
    let notice = tokio::time::timeout(std::time::Duration::from_secs(5), notices.recv())
        .await
        .expect("the detached sibling must report")
        .expect("the notifier stays installed");
    assert!(notice.contains("finished in the background"), "{notice}");
    assert!(notice.contains("sibling report"), "{notice}");
}

/// A reserved floor moves the next child id past it, so a new process never
/// reissues an id an older one already persisted.
#[tokio::test]
async fn reserved_ids_are_never_reissued() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("done");
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(spawner(provider));
    agent_kernel::tools::delegate::reserve_child_ids(5_000_000);
    let report = (agent_kernel::tools::delegate::spec().exec)(
        serde_json::json!({ "task": "go" }),
        Arc::new(ctx),
    )
    .await
    .expect("the child finishes");
    assert!(child_id_of(&report.content) > 5_000_000, "{}", report.content);
}
