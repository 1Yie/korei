//! UI channel setup — one ordered event queue per session, plus the command
//! channel between the shell and the actor.
//!
//! ```text
//! UI ──command───▶ SessionActor ──sampling──▶ SamplerActor (agent-llm)
//! UI ◀─UiEvent───── SessionActor ◀─chunks────
//! ```
//!
//! Deltas may be coalesced or dropped under pressure; control events
//! (`StateChanged`, `ToolCall*`, `ApprovalRequested`, `AssistantMessage`, …)
//! never are. Both kinds share one queue, so a tool capsule cannot overtake
//! the prose that preceded it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tokio::sync::mpsc;

use agent_ipc::UiEvent;
use agent_llm_core::StreamChunk;

/// Internal sampler→session feedback — the kernel-internal counterpart that
/// carries `StreamChunk` plus sampler policy notices. NOT on the wire.
#[derive(Debug)]
pub enum AgentEvent {
    /// One normalized chunk from the provider.
    Chunk(StreamChunk),
    /// Sampler policy event (retry/degrade) → map to `UiEvent::SystemMessage`.
    SamplerNotice(String),
}

/// Delta watermark — once the queue holds this many pending events, incoming
/// deltas are merged into the queue tail (or dropped) instead of growing it.
/// It bounds *delta* memory only: control events are always enqueued, so the
/// queue may exceed this briefly at turn edges. 8192 is ~16 minutes of
/// provider output at the engine's ~120 char/event coalescing — a shell that
/// lags that far behind has bigger problems than this queue.
pub const UI_CHANNEL_CAP: usize = 8192;
pub const CMD_CHANNEL_CAP: usize = 32;

/// Delivery counters for one session's UI queue. Cheap atomics only — the
/// engine sits on the hot path, so nothing here allocates or locks.
#[derive(Debug, Default)]
pub struct UiStats {
    /// Events enqueued as new entries (post-coalesce).
    sent: AtomicU64,
    /// Incoming deltas merged into the queue tail instead of being dropped —
    /// text is preserved, only the event count shrinks.
    coalesced: AtomicU64,
    /// Incoming deltas dropped because the queue was at the watermark and its
    /// tail was a control event. Non-zero means the shell is behind AND a
    /// delta could not be merged, i.e. text was actually lost.
    dropped: AtomicU64,
    /// High-water mark of the queue length.
    depth_peak: AtomicUsize,
}

/// Serialized view of [`UiStats`] — aggregated by `SessionManager::ui_stats`
/// and surfaced on the shell's status bar.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub struct UiStatsSnapshot {
    pub sent: u64,
    pub coalesced: u64,
    pub dropped: u64,
    pub depth_peak: usize,
}

impl UiStatsSnapshot {
    /// Fold another session's counters into this one.
    pub fn add(&mut self, other: &UiStatsSnapshot) {
        self.sent += other.sent;
        self.coalesced += other.coalesced;
        self.dropped += other.dropped;
        self.depth_peak = self.depth_peak.max(other.depth_peak);
    }
}

/// Why an event did not reach the UI. `Closed` is a dead receiver (session
/// torn down / headless context); `Saturated` is a *delta* that had nothing to
/// merge into. Control events never produce `Saturated`.
#[derive(Debug)]
pub enum SendError {
    Closed(UiEvent),
    Saturated(UiEvent),
}

impl SendError {
    /// True when the receiver is gone — the only failure mode callers can act
    /// on (see `AskChannel::ask`, which refuses instead of parking forever).
    pub fn is_closed(&self) -> bool {
        matches!(self, SendError::Closed(_))
    }
}

struct UiQueue {
    queue: Mutex<VecDeque<UiEvent>>,
    /// Capacity-1 doorbell. Producers ring it after enqueuing; a full bell
    /// just means a wake-up is already pending, so the consumer cannot miss
    /// an entry the way a bare `Notify` race would allow.
    bell: mpsc::Sender<()>,
    closed: AtomicBool,
    stats: UiStats,
}

/// Producer half of a session's UI queue. Cheap to clone — every clone shares
/// one ordered queue + one counter block, so `EngineIo`, `ToolCtx`, and the
/// actor all feed (and can report on) the same stream.
#[derive(Clone)]
pub struct UiSink {
    inner: Arc<UiQueue>,
}

/// Consumer half. `recv()` yields events in enqueue order; when the sink is
/// gone (or `close()`d) it drains what is left and then returns `None`.
pub struct UiReceiver {
    inner: Arc<UiQueue>,
    bell: mpsc::Receiver<()>,
    /// Set once the doorbell channel reports every sender gone.
    bell_closed: bool,
}

impl UiSink {
    /// Create a connected sink/receiver pair.
    pub fn channel() -> (UiSink, UiReceiver) {
        let (bell, bell_rx) = mpsc::channel::<()>(1);
        let inner = Arc::new(UiQueue {
            queue: Mutex::new(VecDeque::new()),
            bell,
            closed: AtomicBool::new(false),
            stats: UiStats::default(),
        });
        (
            UiSink {
                inner: inner.clone(),
            },
            UiReceiver {
                inner,
                bell: bell_rx,
                bell_closed: false,
            },
        )
    }

    /// Enqueue one event. Deltas past the watermark are merged into the queue
    /// tail (or dropped, counted); everything else is enqueued unconditionally
    /// so the state machine always reaches the UI.
    pub fn send(&self, ev: UiEvent) -> Result<(), SendError> {
        if self.inner.closed.load(Ordering::Relaxed) {
            return Err(SendError::Closed(ev));
        }
        let mut queue = self.inner.queue.lock().unwrap_or_else(|e| e.into_inner());
        let under_watermark = queue.len() < UI_CHANNEL_CAP;
        match ev {
            // Deltas merge only with a tail from the *same* stream: a delegated
            // child's text must never be appended to the turn's draft (or to
            // another child's) just because both happen to be deltas.
            UiEvent::TextDelta {
                text,
                parent,
                child_id,
            } => {
                if under_watermark {
                    queue.push_back(UiEvent::TextDelta {
                        text,
                        parent,
                        child_id,
                    });
                    self.inner.stats.sent.fetch_add(1, Ordering::Relaxed);
                } else if let Some(UiEvent::TextDelta {
                    text: tail,
                    parent: tail_parent,
                    child_id: tail_child,
                }) = queue.back_mut()
                {
                    if tail_parent.as_deref() != parent.as_deref() || *tail_child != child_id {
                        self.inner.stats.dropped.fetch_add(1, Ordering::Relaxed);
                        return Err(SendError::Saturated(UiEvent::TextDelta {
                            text,
                            parent,
                            child_id,
                        }));
                    }
                    tail.push_str(&text);
                    self.inner.stats.coalesced.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.inner.stats.dropped.fetch_add(1, Ordering::Relaxed);
                    return Err(SendError::Saturated(UiEvent::TextDelta {
                        text,
                        parent,
                        child_id,
                    }));
                }
            }
            UiEvent::ReasoningDelta {
                text,
                parent,
                child_id,
            } => {
                if under_watermark {
                    queue.push_back(UiEvent::ReasoningDelta {
                        text,
                        parent,
                        child_id,
                    });
                    self.inner.stats.sent.fetch_add(1, Ordering::Relaxed);
                } else if let Some(UiEvent::ReasoningDelta {
                    text: tail,
                    parent: tail_parent,
                    child_id: tail_child,
                }) = queue.back_mut()
                {
                    if tail_parent.as_deref() != parent.as_deref() || *tail_child != child_id {
                        self.inner.stats.dropped.fetch_add(1, Ordering::Relaxed);
                        return Err(SendError::Saturated(UiEvent::ReasoningDelta {
                            text,
                            parent,
                            child_id,
                        }));
                    }
                    tail.push_str(&text);
                    self.inner.stats.coalesced.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.inner.stats.dropped.fetch_add(1, Ordering::Relaxed);
                    return Err(SendError::Saturated(UiEvent::ReasoningDelta {
                        text,
                        parent,
                        child_id,
                    }));
                }
            }
            ev => {
                queue.push_back(ev);
                self.inner.stats.sent.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.inner
            .stats
            .depth_peak
            .fetch_max(queue.len(), Ordering::Relaxed);
        drop(queue);
        // A full bell means a wake-up is already queued for the consumer.
        let _ = self.inner.bell.try_send(());
        Ok(())
    }

    /// Mark the stream finished — sends fail afterwards and a parked consumer
    /// wakes up to drain the tail.
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::Relaxed);
        let _ = self.inner.bell.try_send(());
    }

    /// A copy of this queue's counters.
    pub fn stats(&self) -> UiStatsSnapshot {
        snapshot(&self.inner.stats)
    }
}

impl UiReceiver {
    /// Next event in enqueue order, or `None` once the sink is gone *and* the
    /// queue is drained.
    pub async fn recv(&mut self) -> Option<UiEvent> {
        loop {
            if let Some(ev) = self.pop() {
                return Some(ev);
            }
            if self.inner.closed.load(Ordering::Relaxed) || self.bell_closed {
                return None;
            }
            match self.bell.recv().await {
                Some(()) => {}
                // Every sender dropped: drain whatever landed, then stop.
                None => self.bell_closed = true,
            }
        }
    }

    /// Non-blocking `recv` — `Some` when an event is already queued.
    pub fn try_recv(&mut self) -> Option<UiEvent> {
        self.pop()
    }

    /// Stop accepting new events (senders start erroring) while still
    /// draining what is already queued. Used by teardown and by tests that
    /// want "everything emitted so far".
    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::Relaxed);
        let _ = self.inner.bell.try_send(());
    }

    /// A copy of this queue's counters.
    pub fn stats(&self) -> UiStatsSnapshot {
        snapshot(&self.inner.stats)
    }

    fn pop(&mut self) -> Option<UiEvent> {
        self.inner
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
    }
}

impl Drop for UiReceiver {
    /// Dropping the consumer is a hard close: without it, a `UiSink` kept
    /// alive by a headless engine (subagent) would buffer a queue nobody can
    /// ever read.
    fn drop(&mut self) {
        self.close();
    }
}

fn snapshot(stats: &UiStats) -> UiStatsSnapshot {
    UiStatsSnapshot {
        sent: stats.sent.load(Ordering::Relaxed),
        coalesced: stats.coalesced.load(Ordering::Relaxed),
        dropped: stats.dropped.load(Ordering::Relaxed),
        depth_peak: stats.depth_peak.load(Ordering::Relaxed),
    }
}

/// UI-facing endpoints created by [`ui_channels`]. `C` is the host's command
/// type; the kernel never inspects it.
pub struct UiChannels<C> {
    /// UI → session commands.
    pub cmd_rx: mpsc::Receiver<C>,
    /// Kernel → UI events.
    pub event_tx: UiSink,
    /// The frontend's send/recv halves (kept for the bridge layer).
    pub cmd_tx: mpsc::Sender<C>,
    pub event_rx: UiReceiver,
}

/// Create the UI channel pair. The bridge splits `UiChannels` into
/// its command-sender and event-receiver halves.
pub fn ui_channels<C>() -> UiChannels<C> {
    let (cmd_tx, cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
    let (event_tx, event_rx) = UiSink::channel();
    UiChannels {
        cmd_rx,
        event_tx,
        cmd_tx,
        event_rx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(s: &str) -> UiEvent {
        UiEvent::TextDelta {
            text: s.into(),
            parent: None,
            child_id: None,
        }
    }

    /// Fill the queue to the delta watermark.
    fn saturate(tx: &UiSink) {
        for _ in 0..UI_CHANNEL_CAP {
            tx.send(delta("x")).expect("under watermark");
        }
    }

    fn drain(rx: &mut UiReceiver) -> Vec<UiEvent> {
        rx.close();
        let mut out = Vec::new();
        while let Some(ev) = rx.try_recv() {
            out.push(ev);
        }
        out
    }

    #[tokio::test]
    async fn deltas_coalesce_at_the_watermark() {
        let (tx, mut rx) = UiSink::channel();
        saturate(&tx);
        // Past the watermark a delta must merge into the tail, not vanish.
        tx.send(delta("y")).expect("merged");
        let events = drain(&mut rx);
        assert_eq!(events.len(), UI_CHANNEL_CAP);
        let UiEvent::TextDelta { text: tail, .. } = events.last().unwrap() else {
            panic!("expected a text delta tail");
        };
        assert_eq!(tail, "xy");
        let stats = rx.stats();
        assert_eq!(stats.coalesced, 1);
        assert_eq!(stats.dropped, 0, "coalescing must not lose text");
    }

    /// The whole point of the sink: a saturated queue full of deltas still
    /// delivers every control event.
    #[tokio::test]
    async fn control_events_are_never_dropped() {
        let (tx, mut rx) = UiSink::channel();
        saturate(&tx);
        for i in 0..10 {
            tx.send(UiEvent::StateChanged(agent_ipc::AgentState::Finished))
                .unwrap_or_else(|_| panic!("control event {i} dropped"));
        }
        tx.send(UiEvent::AssistantMessage("done".into()))
            .expect("final text");
        // A delta arriving while the tail is a control event cannot merge.
        assert!(matches!(
            tx.send(delta("late")),
            Err(SendError::Saturated(_))
        ));
        let events = drain(&mut rx);
        let control = events
            .iter()
            .filter(|e| !matches!(e, UiEvent::TextDelta { .. }))
            .count();
        assert_eq!(control, 11);
        assert!(matches!(events.last().unwrap(), UiEvent::AssistantMessage(t) if t == "done"));
        assert_eq!(rx.stats().dropped, 1);
    }

    #[tokio::test]
    async fn order_is_preserved_across_deltas_and_control() {
        let (tx, mut rx) = UiSink::channel();
        tx.send(delta("a")).unwrap();
        tx.send(UiEvent::ToolCallStarted {
            name: "read".into(),
            args_preview: "x.rs".into(),
            parent: None,
            child_id: None,
        })
        .unwrap();
        tx.send(delta("b")).unwrap();
        tx.send(UiEvent::StateChanged(agent_ipc::AgentState::Finished))
            .unwrap();
        let events = drain(&mut rx);
        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], UiEvent::TextDelta { text: t, .. } if t == "a"));
        assert!(matches!(&events[1], UiEvent::ToolCallStarted { name, .. } if name == "read"));
        assert!(matches!(&events[2], UiEvent::TextDelta { text: t, .. } if t == "b"));
        assert!(matches!(&events[3], UiEvent::StateChanged(_)));
    }

    #[tokio::test]
    async fn close_drains_then_ends_the_stream() {
        let (tx, mut rx) = UiSink::channel();
        tx.send(delta("a")).unwrap();
        tx.close();
        assert!(matches!(tx.send(delta("b")), Err(SendError::Closed(_))));
        assert!(tx.send(UiEvent::Error("e".into())).is_err());
        assert!(rx.recv().await.is_some(), "queued events still drain");
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn dropping_the_receiver_closes_the_sink() {
        let (tx, rx) = UiSink::channel();
        drop(rx);
        // A headless engine (subagent) must fail fast instead of buffering.
        assert!(matches!(tx.send(delta("a")), Err(SendError::Closed(_))));
    }

    #[tokio::test]
    async fn recv_wakes_on_a_late_send() {
        let (tx, mut rx) = UiSink::channel();
        let handle = tokio::spawn(async move { rx.recv().await });
        tokio::task::yield_now().await;
        tx.send(delta("late")).unwrap();
        let ev = handle.await.unwrap().expect("woken by the doorbell");
        assert!(matches!(ev, UiEvent::TextDelta { text: t, .. } if t == "late"));
    }

    /// A delegated child's delta must not merge into the turn's draft tail (or
    /// into another child's) just because both are `TextDelta` — the parent tag
    /// is part of the merge key.
    #[tokio::test]
    async fn deltas_merge_only_within_the_same_parent() {
        let child = |s: &str| UiEvent::TextDelta {
            text: s.into(),
            parent: Some("subagent #1".into()),
            child_id: Some(1),
        };
        let (tx, mut rx) = UiSink::channel();
        saturate(&tx); // saturated; the tail is the turn's own delta

        // The child's text cannot append to the turn's tail: dropped, not merged.
        assert!(matches!(tx.send(child("c1")), Err(SendError::Saturated(_))));
        assert_eq!(rx.stats().dropped, 1);

        // Make room, seed the child's tail, then saturate on that tail: now the
        // merge key matches, so the child's next deltas coalesce.
        assert!(rx.try_recv().is_some());
        tx.send(child("c1"))
            .expect("room for the child's first delta");
        tx.send(child("c2")).expect("same parent merges");
        tx.send(child("c3")).expect("same parent merges");
        tx.send(delta("turn"))
            .expect_err("turn text must not merge into a child's tail");

        let events = drain(&mut rx);
        let UiEvent::TextDelta {
            text,
            parent,
            child_id,
        } = events.last().unwrap()
        else {
            panic!("expected the child's delta as the tail: {events:?}");
        };
        assert_eq!(parent.as_deref(), Some("subagent #1"));
        assert_eq!(*child_id, Some(1));
        assert!(text.ends_with("c1c2c3"), "{text}");
        assert_eq!(rx.stats().coalesced, 2);
    }
}
