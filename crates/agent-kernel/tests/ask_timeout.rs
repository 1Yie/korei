//! `ask_question` must not park the turn on a UI that never answers: the
//! deadline hands the model a timeout result, the same question is not shown
//! twice, a cancel stays an error, and an answer that lands too late is
//! refused instead of silently winning.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_ipc::events::UiEvent;
use agent_kernel::channels::{UiReceiver, UiSink};
use agent_kernel::tools::host_slots::AskChannel;
use agent_kernel::tools::registry::ToolCtx;

fn question_args(question: &str) -> serde_json::Value {
    serde_json::json!({
        "question": question,
        "options": [{ "label": "korei" }, { "label": "other" }]
    })
}

/// One interactive question channel with a short deadline.
fn ask_ctx(timeout: Duration) -> (Arc<ToolCtx>, Arc<AskChannel>, UiReceiver) {
    let (ui_tx, ui_rx) = UiSink::channel();
    let ask = Arc::new(AskChannel::new(Some(ui_tx)));
    ask.set_timeout(timeout);
    let mut ctx = ToolCtx::new();
    ctx.install_ask(ask.clone());
    (Arc::new(ctx), ask, ui_rx)
}

fn drain(rx: &mut UiReceiver) -> Vec<UiEvent> {
    rx.close();
    let mut events = Vec::new();
    while let Some(ev) = rx.try_recv() {
        events.push(ev);
    }
    events
}

fn asked_ids(events: &[UiEvent]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|ev| match ev {
            UiEvent::QuestionAsked { request_id, .. } => Some(*request_id),
            _ => None,
        })
        .collect()
}

fn timed_out_questions(events: &[UiEvent]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|ev| match ev {
            UiEvent::QuestionTimedOut { request_id } => Some(*request_id),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn an_answer_before_the_deadline_is_returned() {
    let (ctx, ask, mut ui_rx) = ask_ctx(Duration::from_secs(30));
    let answerer = ask.clone();
    let (res, id) = tokio::join!(
        (agent_kernel::tools::ask::spec().exec)(question_args("which crate?"), ctx),
        async {
            while let Some(ev) = ui_rx.recv().await {
                if let UiEvent::QuestionAsked { request_id, .. } = ev {
                    assert!(answerer.answer(request_id, "korei".into()), "the card is parked");
                    return request_id;
                }
            }
            panic!("no question card was drawn");
        }
    );
    let res = res.expect("asking is not an error");
    assert!(
        res.content.contains("用户回答：korei（选项 0）"),
        "{}",
        res.content
    );
    // The answer closed the card: a second click resolves nothing.
    assert!(!ask.answer(id, "korei".into()));

    // The answerer consumed the one card it answered; nothing after it is a
    // second card or a timeout.
    let events = drain(&mut ui_rx);
    assert!(asked_ids(&events).is_empty(), "{events:#?}");
    assert!(timed_out_questions(&events).is_empty(), "{events:#?}");
}

#[tokio::test]
async fn an_unanswered_question_times_out_into_a_result() {
    let (ctx, _ask, mut ui_rx) = ask_ctx(Duration::from_millis(150));
    let res = tokio::time::timeout(
        Duration::from_secs(5),
        (agent_kernel::tools::ask::spec().exec)(question_args("which crate?"), ctx),
    )
    .await
    .expect("the question must not park the turn")
    .expect("a timeout is a result, not an error");

    assert!(
        res.content.starts_with("[ask_question timeout]"),
        "{}",
        res.content
    );
    assert!(res.content.contains("within 0.15s"), "{}", res.content);
    assert!(res.content.contains("korei / other"), "{}", res.content);

    let events = drain(&mut ui_rx);
    let ids = asked_ids(&events);
    assert_eq!(ids.len(), 1, "{events:#?}");
    assert_eq!(timed_out_questions(&events), ids, "{events:#?}");
}

#[tokio::test]
async fn an_answer_after_the_deadline_wins_nothing() {
    let (ctx, ask, mut ui_rx) = ask_ctx(Duration::from_millis(150));
    let res = tokio::time::timeout(
        Duration::from_secs(5),
        (agent_kernel::tools::ask::spec().exec)(question_args("which crate?"), ctx),
    )
    .await
    .expect("the question must not park the turn")
    .expect("a timeout is a result, not an error");
    assert!(
        res.content.starts_with("[ask_question timeout]"),
        "{}",
        res.content
    );

    let events = drain(&mut ui_rx);
    let ids = asked_ids(&events);
    assert_eq!(ids.len(), 1, "{events:#?}");
    // The card is already closed — the late text goes nowhere.
    assert!(!ask.answer(ids[0], "korei".into()));
}

#[tokio::test]
async fn a_cancel_is_an_error_and_takes_no_answer() {
    let (ctx, ask, mut ui_rx) = ask_ctx(Duration::from_secs(30));
    let cancel = Arc::new(AtomicBool::new(false));
    let ctx = Arc::new(ctx.with_cancel(cancel.clone()));

    let (res, ()) = tokio::join!(
        (agent_kernel::tools::ask::spec().exec)(question_args("which crate?"), ctx),
        async {
            tokio::time::sleep(Duration::from_millis(150)).await;
            cancel.store(true, Ordering::Relaxed);
        }
    );
    let err = res.expect_err("a cancelled question is an error");
    assert!(err.to_string().contains("cancelled"), "{err}");

    let events = drain(&mut ui_rx);
    let ids = asked_ids(&events);
    assert_eq!(ids.len(), 1, "{events:#?}");
    assert!(timed_out_questions(&events).is_empty(), "{events:#?}");
    assert!(
        !ask.answer(ids[0], "korei".into()),
        "a cancelled card takes no answer"
    );
}

#[tokio::test]
async fn an_identical_question_after_a_timeout_is_not_shown_again() {
    let (ctx, _ask, mut ui_rx) = ask_ctx(Duration::from_millis(120));
    let first = tokio::time::timeout(
        Duration::from_secs(5),
        (agent_kernel::tools::ask::spec().exec)(question_args("Which  CRATE?"), ctx.clone()),
    )
    .await
    .expect("the question must not park the turn")
    .expect("a timeout is a result, not an error");
    assert!(
        first.content.starts_with("[ask_question timeout]"),
        "{}",
        first.content
    );
    assert!(
        !first.content.contains("not shown again"),
        "{}",
        first.content
    );

    // Same question, different spacing and case: still the same question.
    let second = tokio::time::timeout(
        Duration::from_secs(5),
        (agent_kernel::tools::ask::spec().exec)(question_args("  which crate?  "), ctx),
    )
    .await
    .expect("the repeat answers immediately instead of drawing a card")
    .expect("a timeout is a result, not an error");
    assert!(
        second
            .content
            .contains("The same question already timed out; it was not shown again."),
        "{}",
        second.content
    );

    let events = drain(&mut ui_rx);
    assert_eq!(asked_ids(&events).len(), 1, "only one card: {events:#?}");
    assert_eq!(timed_out_questions(&events).len(), 1, "{events:#?}");
}
