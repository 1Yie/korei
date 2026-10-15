//! `compaction` — history summarization near the context window.
//!
//! - Trigger: estimated tokens ≈80% of `context_window` (user-settable
//!   70/80/90%), corrected upward by the provider's last reported
//!   `prompt_tokens` when available. Compaction runs inline at trigger time.
//! - One summarization pass: the prefix (everything before the ~25%
//!   suffix) compresses into a single `NOTE`; a prior NOTE folds into the
//!   new one rather than stacking.
//! - The sanitize pipeline runs before every sample, not just at compaction.
//! - A failed compaction suppresses further triggers until one succeeds —
//!   with an emergency escape once the estimate nears the window edge.
//!
//! Token estimate: ASCII ≈4 chars/token, non-ASCII ≈1 token/char — the
//! provider's actual `prompt_tokens` overrides it whenever reported.

use agent_llm_core::types::{ChatMessage, Role};

/// What one completed compaction pass did — drives the UI card and the
/// session's post-compaction accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionOutcome {
    /// Estimated tokens across the history before the splice.
    pub before_tokens: usize,
    /// Estimated tokens after the splice (the note + verbatim suffix).
    pub after_tokens: usize,
    /// Messages folded into the note — `prefix_end - 1`, because
    /// `history[0]` (the kernel system prompt) is never compacted.
    pub removed_messages: usize,
    /// The raw summary the compactor produced (unwrapped — `apply` frames
    /// it for the wire).
    pub note: String,
}

/// Fraction of `context_window` that triggers compaction.
pub const COMPACT_AT: f32 = 0.80;
/// Fill fraction at which a latched `UntilSuccess` suppression is
/// overridden — a transient compactor error must not strand the session
/// in silent truncation when the window is nearly full.
pub const EMERGENCY_COMPACT_AT: f32 = 0.95;

/// Rough token estimate — ASCII ≈4 chars/token, non-ASCII (CJK, emoji)
/// ≈1 token/char. `String::len()` is BYTES, so `len/4` alone reads ~0.75
/// tokens per CJK char against a real cost of ~1-1.5 — underestimating on
/// the trigger side. The provider's reported `prompt_tokens` supersedes
/// this whenever available (see `Engine::last_prompt_tokens`).
pub fn estimate_tokens(msgs: &[ChatMessage]) -> usize {
    msgs.iter().map(msg_token_est).sum()
}

/// One message's share of the estimate — content + tool-call payload,
/// per-message overhead, and a flat ~1100/image vision cost.
fn msg_token_est(m: &ChatMessage) -> usize {
    let c = m.content.as_deref().map(est_text).unwrap_or(0);
    let tc = m
        .tool_calls
        .as_ref()
        .map(|v| {
            v.iter()
                .map(|t| est_text(&t.name) + est_text(&t.arguments))
                .sum::<usize>()
        })
        .unwrap_or(0);
    c + tc + 4 + m.images.len() * 1100 // +4 msg overhead; ~1100/image
}

/// ASCII chars count 1/4 token each; every non-ASCII char counts 1 —
/// summing numerators over 4 keeps the math in integers.
fn est_text(s: &str) -> usize {
    s.chars()
        .map(|c| if c.is_ascii() { 1usize } else { 4usize })
        .sum::<usize>()
        / 4
}

/// Should compaction fire at a caller-chosen fraction? The settings UI
/// exposes 70/80/90% of the window.
pub fn should_compact_at(tokens: usize, window: usize, frac: f32) -> bool {
    tokens >= (window as f32 * frac) as usize
}

/// Should compaction fire? `tokens` is the current estimate, `window` the
/// model's context size.
pub fn should_compact(tokens: usize, window: usize) -> bool {
    should_compact_at(tokens, window, COMPACT_AT)
}

/// How many of the newest tool-result frames (screenshots) stay on the wire.
/// A computer-use loop takes a fresh screenshot after every action and each
/// one is re-encoded into every later request — without a bound a long loop
/// ships dozens of stale desktop frames the model no longer needs. The newest
/// frame is the one coordinates are read from; one extra gives it the
/// before/after pair that shows what the last action changed.
pub const KEEP_TOOL_IMAGES: usize = 2;

/// Drop the image off every tool result except the newest `keep` frames,
/// leaving a one-line note so the text still says why the picture is gone.
/// Only `Role::Tool` rows are touched — images the user attached stay.
/// Idempotent: a row already stripped has no images and is skipped.
pub fn prune_stale_tool_images(history: &mut [ChatMessage], keep: usize) {
    let mut seen = 0usize;
    for m in history.iter_mut().rev() {
        if m.role != Role::Tool || m.images.is_empty() {
            continue;
        }
        seen += 1;
        if seen > keep {
            m.images.clear();
            if let Some(c) = m.content.as_mut() {
                c.push_str("\n[image dropped: superseded by a newer image]");
            }
        }
    }
}

/// Sanitize + budget-fit the history in place; runs before every sample.
///
/// Tool calls are flattened into text only for providers without native tool
/// calls — flattening orphans every `Role::Tool` result (P1-b). Reasoning traces
/// are left alone: display-only, and a reopened session replays them. Image refs
/// survive only when the model declares image input, the marker text otherwise.
/// Oldest non-system messages are dropped until the estimate fits `budget_tokens`.
pub fn sanitize_for_sample(
    history: &mut Vec<ChatMessage>,
    budget_tokens: usize,
    keep_images: bool,
) {
    // (2) reasoning stays on the message. It is a *display* field: the wire
    // bodies are built from role/content/tool_calls only, and `openai_completions`
    // (the one adapter that serializes the whole struct) strips it. Dropping
    // it here would silently empty every replayed thinking block on the next
    // snapshot — sanitize runs against the real history, not a copy.

    // (3) images ride the `images` field now — real parts for a vision
    // model, dropped for a text-only one (the `<attached-image>` marker
    // in `content` keeps the path reference either way).
    if !keep_images {
        for m in history.iter_mut() {
            m.images.clear();
        }
    } else {
        prune_stale_tool_images(history, KEEP_TOOL_IMAGES);
    }

    // (4) fit to budget — drop oldest non-system messages.
    fit_conversation_to_budget(history, budget_tokens);
}

/// Flatten `tool_calls` → `[call: name(args)]` text — the wire shape for
/// text-protocol providers (they can't read a structured `tool_calls`
/// array). **Wire-only**: apply on the request clone, never on the live
/// history — the persisted form keeps structured `tool_calls` so the
/// snapshot/replay keeps real call structure and `strip_call_echo`
/// never has to choose between "intentional flatten" and "provider
/// echo leaked into content" (which was the same text, indistinguishable).
pub fn flatten_tool_calls_for_wire(history: &mut Vec<ChatMessage>) {
    for m in history.iter_mut() {
        if let Some(calls) = m.tool_calls.take() {
            let flat: String = calls
                .iter()
                .map(|c| format!("\n[call: {}({})]", c.name, abbreviate(&c.arguments, 200)))
                .collect();
            let mut c = m.content.take().unwrap_or_default();
            c.push_str(&flat);
            m.content = Some(c);
        }
    }
}

/// Drop oldest non-system messages until `estimate_tokens(history)` fits
/// `budget`. The system prompt (history[0]) is never dropped.
///
/// Tool-pair integrity (Responses API): dropping a `Role::Tool` row would
/// orphan its `function_call` — strict backends (deepseek) reject the
/// request with `No tool output found for tool call …`. So removing a tool
/// row also strips the matching `tool_calls` entry on its assistant row;
/// an assistant row left with no calls AND no content is removed too (a
/// blank turn replays as a meaningless message).
fn fit_conversation_to_budget(history: &mut Vec<ChatMessage>, budget: usize) {
    while history.len() > 1 && estimate_tokens(history) > budget {
        // Find the first non-system index and remove it — prefer dropping
        // tool results (they're the bulkiest) then assistant turns.
        let drop_idx = history
            .iter()
            .enumerate()
            .skip(1)
            .position(|(_, m)| m.role == Role::Tool)
            .map(|p| p + 1)
            .or_else(|| {
                history
                    .iter()
                    .enumerate()
                    .skip(1)
                    .position(|(_, m)| m.role != Role::System)
                    .map(|p| p + 1)
            });
        match drop_idx {
            Some(i) if i < history.len() => {
                let removed = history.remove(i);
                if removed.role == Role::Tool {
                    if let Some(tid) = &removed.tool_call_id {
                        // Orphan guard: the matching assistant row's call
                        // is pulled with its output so the pair stays
                        // atomic — an unpaired function_call is a 400.
                        for m in history.iter_mut() {
                            if m.role != Role::Assistant {
                                continue;
                            }
                            if let Some(calls) = &mut m.tool_calls {
                                let before = calls.len();
                                calls.retain(|c| c.id != *tid);
                                if calls.len() != before {
                                    // Assistant row now carries neither
                                    // calls nor text → drop the blank turn.
                                    if calls.is_empty()
                                        && m.content
                                            .as_deref()
                                            .map(|s| s.trim().is_empty())
                                            .unwrap_or(true)
                                    {
                                        m.tool_calls = None;
                                        m.content = None;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                // Sweep any assistant row that ended up fully empty
                // (no calls, no content) — it replays as a blank message.
                history.retain(|m| {
                    m.role != Role::Assistant
                        || m.content
                            .as_deref()
                            .map(|s| !s.trim().is_empty())
                            .unwrap_or(false)
                        || m.tool_calls
                            .as_ref()
                            .map(|c| !c.is_empty())
                            .unwrap_or(false)
                });
            }
            _ => break, // only system left — can't shrink further
        }
    }
}

/// The compaction plan — prefix/suffix split for the two-pass summarize.
#[derive(Debug)]
pub struct CompactionPlan {
    /// history[0..cut) → summarized into `NOTE₁`.
    pub prefix_end: usize,
    /// Estimated tokens in the prefix (the part being compressed away).
    pub prefix_tokens: usize,
}

/// Decide where to split. The suffix keeps the last N messages verbatim so
/// the model doesn't lose the immediate task context — ~25% of the window.
pub fn plan(history: &[ChatMessage], window: usize) -> Option<CompactionPlan> {
    if history.len() < 4 {
        return None; // nothing worth compacting
    }
    // Keep the system prompt + the most recent ~25% of the window's worth
    // of messages as the suffix. Walk back from the end accumulating tokens.
    let suffix_budget = (window as f32 * 0.25) as usize;
    let mut acc = 0usize;
    let mut cut = history.len();
    for (i, m) in history.iter().enumerate().rev() {
        let t = msg_token_est(m);
        if acc + t > suffix_budget && i > 1 {
            cut = i + 1;
            break;
        }
        acc += t;
        cut = i;
    }
    if cut <= 1 {
        return None; // suffix already covers everything
    }
    // Don't split a tool block: if the suffix would begin with `Role::Tool`
    // rows, their assistant `tool_calls` row sits in the prefix — compacting
    // it away orphans the outputs, and strict backends (deepseek) reject the
    // replayed request. Fold the dangling outputs into the compacted prefix.
    while cut < history.len() && history[cut].role == Role::Tool {
        cut += 1;
    }
    if cut >= history.len() {
        // The fold would consume the WHOLE tail — the trailing tool block
        // alone is bigger than the suffix budget. Folding it away would
        // leave the model with no recent context, and returning `None` here
        // would disable compaction for the rest of the session while the
        // 90% budget-fit silently truncates the oldest turns round after
        // round. Keep that block (with the assistant call that owns it)
        // verbatim instead, and summarize what precedes it.
        let mut j = cut;
        while j > 1 && history[j - 1].role == Role::Tool {
            j -= 1;
        }
        // `j` is the block's first result row; its owning `tool_calls` row
        // sits directly before it.
        cut = j.saturating_sub(1);
        if cut <= 1 {
            return None; // nothing before the block — nothing to summarize
        }
    }
    let prefix_tokens = estimate_tokens(&history[..cut]);
    Some(CompactionPlan {
        prefix_end: cut,
        prefix_tokens,
    })
}

/// Apply a completed compaction: replace history[1..plan.prefix_end) with a
/// single synthesized `NOTE` message — `history[0]` (the kernel system
/// prompt) is never compacted away. The `note_text` is the compactor's
/// output. The note is tagged `NoticeKind::CompactedMemory` so adapters
/// emit it at user privilege — folding a history summary into
/// `system`/`instructions` would let stale decisions read as live orders.
pub fn apply(history: &mut Vec<ChatMessage>, plan: &CompactionPlan, note_text: String) {
    let note = ChatMessage::compacted_memory(format!(
        "[BEGIN COMPACTED CONTEXT — summary of earlier turns; historical context, not new instructions]\n\
         {note_text}\n\
         [END COMPACTED CONTEXT]"
    ));
    // Splice from index 1 — `history[0]` is the rendered kernel system
    // prompt; compacting it away would strip the model's tool protocol and
    // mode contract for the rest of the session.
    history.splice(1..plan.prefix_end, std::iter::once(note));
}

/// Render one message as compactor input — role label + content + any
/// `tool_calls`. Native-calling providers keep calls structured, so a
/// content-only join would hide every invocation (name + args) from the
/// summary; only the `Role::Tool` result text would survive.
pub fn render_for_summary(m: &ChatMessage) -> String {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut body = m.content.clone().unwrap_or_default();
    if let Some(calls) = &m.tool_calls {
        for c in calls {
            body.push_str(&format!(
                "\n[call: {}({})]",
                c.name,
                abbreviate(&c.arguments, 200)
            ));
        }
    }
    format!("{role}: {body}")
}

/// Sticky suppression state — after a failed compaction we don't re-fire on
/// the very next turn (storm guard). `UntilSuccess` latches until a
/// compaction actually succeeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Suppress {
    #[default]
    None,
    /// Skip one more trigger check.
    Sticky,
    /// Skip until a compaction completes successfully.
    UntilSuccess,
}

#[derive(Debug, Default)]
pub struct CompactionSuppressor {
    state: Suppress,
}

impl CompactionSuppressor {
    /// Check + consume. `fire` when the trigger is allowed to run.
    pub fn check(&mut self) -> bool {
        self.check_at_fill(0.0)
    }

    /// Same as `check`, but `fill` is the current estimated fraction of
    /// the context window — a latched `UntilSuccess` is overridden once
    /// the window is nearly full, so one transient provider error can't
    /// strand the session in silent oldest-first truncation.
    pub fn check_at_fill(&mut self, fill: f32) -> bool {
        match self.state {
            Suppress::None => true,
            Suppress::Sticky => {
                self.state = Suppress::None;
                false
            }
            Suppress::UntilSuccess => fill >= EMERGENCY_COMPACT_AT,
        }
    }

    /// Compaction failed — latch suppression so the next turn doesn't retry.
    pub fn on_failure(&mut self) {
        self.state = Suppress::UntilSuccess;
    }

    /// Compaction succeeded — clear all suppression.
    pub fn on_success(&mut self) {
        self.state = Suppress::None;
    }
}

/// Truncate `s` to `n` chars at a char boundary (for flattened tool args).
fn abbreviate(s: &str, n: usize) -> &str {
    if s.len() <= n {
        s
    } else {
        let mut end = n;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_llm_core::types::NoticeKind;

    fn msg(role: Role, text: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: None,
            ts: None,
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
        }
    }

    #[test]
    fn estimate_is_chars_over_four() {
        let h = vec![msg(Role::User, &"a".repeat(400))];
        // 400 chars / 4 + 4 overhead = 104
        assert_eq!(estimate_tokens(&h), 104);
    }

    #[test]
    fn estimate_counts_cjk_at_one_token_per_char() {
        // 100 CJK chars ≈100 real tokens — `len()/4` would read ~75 and
        // delay the trigger past the true fill.
        let h = vec![msg(Role::User, &"汉".repeat(100))];
        assert_eq!(estimate_tokens(&h), 104);
    }

    #[test]
    fn trigger_at_80_percent() {
        assert!(should_compact(80_000, 100_000));
        assert!(!should_compact(79_000, 100_000));
    }

    #[test]
    fn sanitize_flattens_tool_calls() {
        let mut h = vec![
            msg(Role::System, "sys"),
            ChatMessage {
                role: Role::Assistant,
                content: Some("calling".into()),
                tool_calls: Some(vec![agent_llm_core::types::ToolCall {
                    id: "1".into(),
                    name: "list_dir".into(),
                    arguments: "{\"path\":\".\"}".into(),
                }]),
                tool_call_id: None,
                is_error: None,
                notice: None,
                ts: None,
                images: Vec::new(),

                reasoning: None,
                duration_ms: None,
            },
        ];
        // text-protocol provider → tool_calls flatten into message text on
        // the wire-only copy (the persisted history keeps structure).
        flatten_tool_calls_for_wire(&mut h);
        assert!(h[1].tool_calls.is_none());
        assert!(h[1].content.as_deref().unwrap().contains("[call: list_dir"));

        // native provider path → no flatten call — tool_calls stay
        // structured, else the matching Role::Tool result orphans (P1-b).
        let mut h2 = vec![
            msg(Role::System, "sys"),
            ChatMessage {
                role: Role::Assistant,
                content: Some("ok".into()),
                tool_calls: Some(vec![agent_llm_core::types::ToolCall {
                    id: "1".into(),
                    name: "list_dir".into(),
                    arguments: "{\"path\":\".\"}".into(),
                }]),
                tool_call_id: None,
                is_error: None,
                notice: None,
                ts: None,
                images: Vec::new(),

                reasoning: None,
                duration_ms: None,
            },
        ];
        sanitize_for_sample(&mut h2, 100_000, true);
        assert!(
            h2[1].tool_calls.is_some(),
            "native tool_calls must not flatten"
        );
        assert!(!h2[1].content.as_deref().unwrap().contains("[call:"));
    }

    #[test]
    fn fit_drops_tool_results_first() {
        let mut h = vec![
            msg(Role::System, "sys"),
            msg(Role::User, "q"),
            msg(Role::Tool, &"x".repeat(4_000)), // bulky tool result
            msg(Role::Assistant, "a"),
        ];
        // Budget so only ~system + last two survive.
        sanitize_for_sample(&mut h, 200, true);
        assert!(h.iter().all(|m| m.role != Role::Tool));
        assert!(h[0].role == Role::System); // system never dropped
    }

    #[test]
    fn fit_removes_orphaned_tool_call_with_its_output() {
        // A tool result is dropped for budget → the matching `tool_calls`
        // entry on the assistant row must go too, else the next request
        // replays an unpaired `function_call` and deepseek 400s with
        // "No tool output found for tool call …".
        let mut h = vec![
            msg(Role::System, "sys"),
            msg(Role::User, "q"),
            ChatMessage {
                role: Role::Assistant,
                content: Some("calling".into()),
                tool_calls: Some(vec![
                    agent_llm_core::types::ToolCall {
                        id: "call_a".into(),
                        name: "bash".into(),
                        arguments: "{}".into(),
                    },
                    agent_llm_core::types::ToolCall {
                        id: "call_b".into(),
                        name: "bash".into(),
                        arguments: "{}".into(),
                    },
                ]),
                tool_call_id: None,
                is_error: None,
                notice: None,
                ts: None,
                images: Vec::new(),

                reasoning: None,
                duration_ms: None,
            },
            {
                let mut m = msg(Role::Tool, &"output_a ".repeat(500));
                m.tool_call_id = Some("call_a".into());
                m
            },
            {
                let mut m = msg(Role::Tool, "out_b");
                m.tool_call_id = Some("call_b".into());
                m
            },
            msg(Role::Assistant, "done"),
        ];
        // Budget fits everything except the bulky call_a output — one
        // drop removes it and the orphaned call_a entry, nothing else.
        let total = estimate_tokens(&h);
        let call_a_out = 500 * "output_a ".len() / 4 + 4;
        sanitize_for_sample(&mut h, total - call_a_out + 50, true);
        // The bulky call_a output is gone; its call must be gone too.
        let assistant = h
            .iter()
            .find(|m| m.role == Role::Assistant && m.tool_calls.is_some())
            .unwrap();
        let ids: Vec<&str> = assistant
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        assert!(!ids.contains(&"call_a"), "orphaned call_a must be removed");
        assert!(ids.contains(&"call_b"), "call_b survives with its output");
        // call_b's output row is still present.
        assert!(h
            .iter()
            .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("call_b")));
    }

    #[test]
    fn plan_splits_prefix_suffix() {
        let mut h = vec![msg(Role::System, "sys")];
        for _i in 0..20 {
            h.push(msg(Role::User, &"u".repeat(200).repeat(1).to_string()));
            h.push(msg(Role::Assistant, &"a".repeat(200)));
        }
        let p = plan(&h, 1_000).unwrap(); // 25% suffix = ~250 tokens
        assert!(p.prefix_end > 1 && p.prefix_end < h.len());
        // suffix stays under its ~250-token budget
        assert!(estimate_tokens(&h[p.prefix_end..]) <= 300);
    }

    #[test]
    fn plan_keeps_an_over_budget_trailing_tool_block() {
        // A trailing tool block bigger than the whole suffix budget used to
        // make `plan` bail (`None`): compaction then stayed disabled for the
        // rest of the session while the 90% budget-fit silently truncated
        // the oldest turns round after round. The block stays verbatim (with
        // its owning call row) and everything before it is compacted.
        let mut h = vec![msg(Role::System, "sys")];
        h.push(msg(Role::User, &"u".repeat(8_000))); // ≈2000 prefix tokens
        h.push(ChatMessage {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![agent_llm_core::types::ToolCall {
                id: "call_big".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            }]),
            tool_call_id: None,
            is_error: None,
            notice: None,
            ts: None,
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
        });
        h.push(msg(Role::Tool, &"T".repeat(20_000))); // ≈5000 tokens ≫ 25%

        let p = plan(&h, 4_000).expect("an over-budget trailing tool block must still compact");
        assert!(p.prefix_end > 1 && p.prefix_end < h.len());
        // The kept tail starts at the owning call, so the pair never orphans.
        assert_eq!(h[p.prefix_end].role, Role::Assistant);
        assert_eq!(h[p.prefix_end + 1].role, Role::Tool);
        assert!(estimate_tokens(&h[p.prefix_end..]) > 4_000 / 4);
    }

    #[test]
    fn apply_replaces_prefix_with_note() {
        // Enough history that `plan` actually splits — 30 pairs of 200-char
        // messages against a 400-token window (suffix keeps ~100 tokens).
        let mut h = vec![msg(Role::System, "sys")];
        for _ in 0..30 {
            h.push(msg(Role::User, &"u".repeat(200)));
            h.push(msg(Role::Assistant, &"a".repeat(200)));
        }
        let p = plan(&h, 400).unwrap();
        let orig_len = h.len();
        apply(&mut h, &p, "summary".into());
        // Splice starts at 1 — the system prompt survives compaction.
        assert_eq!(h.len(), orig_len - p.prefix_end + 2);
        assert_eq!(h[0].content.as_deref().unwrap(), "sys");
        assert!(h[1]
            .content
            .as_deref()
            .unwrap()
            .contains("[BEGIN COMPACTED CONTEXT"));
        assert!(h[1]
            .content
            .as_deref()
            .unwrap()
            .contains("[END COMPACTED CONTEXT]"));
        assert_eq!(h[1].notice, Some(NoticeKind::CompactedMemory));
    }

    #[test]
    fn suppressor_blocks_retry_storm() {
        let mut s = CompactionSuppressor::default();
        s.on_failure();
        assert!(!s.check()); // UntilSuccess latches
        assert!(!s.check());
        s.on_success();
        assert!(s.check());
    }

    #[test]
    fn suppressor_escapes_at_emergency_fill() {
        let mut s = CompactionSuppressor::default();
        s.on_failure();
        assert!(!s.check_at_fill(0.80)); // still suppressed below the edge
        assert!(s.check_at_fill(0.96)); // emergency — near-full window fires anyway
                                        // The latch survives the emergency check: only a success clears it.
        assert!(!s.check_at_fill(0.80));
    }

    #[test]
    fn render_for_summary_includes_tool_calls() {
        let mut m = msg(Role::Assistant, "checking");
        m.tool_calls = Some(vec![agent_llm_core::types::ToolCall {
            id: "c1".into(),
            name: "fs_read".into(),
            arguments: "{\"path\":\"a.rs\"}".into(),
        }]);
        let r = render_for_summary(&m);
        assert!(r.starts_with("assistant: checking"));
        assert!(r.contains("[call: fs_read("));
    }
    fn frame_row(id: &str) -> ChatMessage {
        let mut m = msg(Role::Tool, "screenshot saved: shot.png");
        m.tool_call_id = Some(id.into());
        m.images = vec![agent_llm_core::types::ImageRef::for_path("/ws/shot.png".into()).unwrap()];
        m
    }

    #[test]
    fn only_the_newest_frames_stay_on_the_wire() {
        let mut h = vec![msg(Role::System, "sys")];
        for i in 0..5 {
            h.push(frame_row(&format!("c{i}")));
        }
        prune_stale_tool_images(&mut h, 2);
        let kept: Vec<bool> = h.iter().skip(1).map(|m| !m.images.is_empty()).collect();
        assert_eq!(kept, [false, false, false, true, true]);
        // The stripped rows say why, once; the kept rows are untouched.
        assert!(h[1].content.as_deref().unwrap().contains("image dropped"));
        assert!(!h[5].content.as_deref().unwrap().contains("image dropped"));
        // Idempotent: a second pass appends nothing.
        let before = h[1].content.clone();
        prune_stale_tool_images(&mut h, 2);
        assert_eq!(h[1].content, before);
    }

    #[test]
    fn user_attached_images_are_never_pruned() {
        let mut user = msg(Role::User, "look at this");
        user.images = vec![agent_llm_core::types::ImageRef::for_path("/ws/a.png".into()).unwrap()];
        let mut h = vec![msg(Role::System, "sys"), user];
        for i in 0..4 {
            h.push(frame_row(&format!("c{i}")));
        }
        prune_stale_tool_images(&mut h, 1);
        assert_eq!(h[1].images.len(), 1, "the user's image must survive");
        assert_eq!(h.iter().filter(|m| !m.images.is_empty()).count(), 2);
    }

    #[test]
    fn sanitize_applies_the_frame_bound_for_vision_models() {
        let mut h = vec![msg(Role::System, "sys")];
        for i in 0..6 {
            h.push(frame_row(&format!("c{i}")));
        }
        sanitize_for_sample(&mut h, 1_000_000, true);
        assert_eq!(
            h.iter().filter(|m| !m.images.is_empty()).count(),
            KEEP_TOOL_IMAGES
        );
    }
}
