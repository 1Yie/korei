//! `batch_execute` — parallel execution for independent read-only calls.
//!
//! Eligibility is by `ToolSpec::class`, not by name: only `Observation` tools
//! qualify (`readonly` alone is not enough), and network items are capped lower
//! than local ones. Results keep input order, per-item failures are reported
//! rather than fatal, and cancellation is batch-scoped so tools that poll
//! `ctx.cancel` actually abort. No permission gate runs: only auto-allowed
//! specs are eligible by construction.
//!
//! `screenshot` is batchable (it is an `Observation`), `computer` is not (it is
//! `Process`). Never pair them in one batch: a `screenshot` item installs the
//! coordinate scale on the shared desktop backend, and a concurrent `computer`
//! in another item reads whichever `last` was written first — sequence the
//! pair across batches (`screenshot` in one turn, `computer` in the next)
//! or the click maps onto the previous frame.

use super::registry::{Args, ToolClass, ToolCtx, ToolError, ToolResult, ToolSpec};
use futures::{FutureExt, StreamExt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_BATCH_CALLS: usize = 16;
const MAX_BATCH_CONCURRENCY: usize = 4;
/// Network reads share the observation pool but get a tighter ceiling —
/// 16 concurrent fetches is an SSRF/rate-limit amplifier, not a feature.
const MAX_NETWORK_CALLS: usize = 4;
const MAX_BATCH_WALL: Duration = Duration::from_secs(30);
const MAX_BATCH_OUTPUT: usize = 5 * 1024 * 1024;

/// The batch's own resource envelope — deadline is a BATCH deadline,
/// not a per-item timeout; every item sees the remaining slice at start.
struct BatchBudget {
    deadline: Instant,
    /// Set by the watchdog when the deadline lands or the parent turn is
    /// cancelled — propagates into every item's ctx so blocking work
    /// (spawn_blocking walks, in-flight fetches) actually stops.
    cancel: Arc<AtomicBool>,
}

#[derive(Debug)]
enum ItemStatus {
    Ok,
    Error,
    Timeout,
    /// Failed eligibility up front (non-observation class / unknown tool).
    Rejected,
    Cancelled,
}

struct ItemOut {
    tool: String,
    status: ItemStatus,
    body: String,
    /// Images an item produced (`screenshot` is batchable by class) — folded
    /// onto the batch result so a batched capture keeps its frame.
    images: Vec<agent_llm_core::types::ImageRef>,
    duration_ms: u64,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct BatchCall {
    /// Tool name — must be an available observation tool. Mutations,
    /// delegation, questions, session state, and control signals are
    /// rejected per-item. `name`/`tool_name` are accepted because models
    /// reach for them.
    #[serde(alias = "name", alias = "tool_name")]
    tool: String,
    /// Arguments for that tool, same shape as a direct call. `arguments` is
    /// accepted as an alias (the MCP spelling).
    #[serde(default, alias = "arguments")]
    args: serde_json::Value,
    /// Argument keys written inline instead of under `args`
    /// (`{"tool":"smart_read","path":"…"}`) — absorbed by [`BatchCall::normalized`].
    #[serde(flatten)]
    inline: serde_json::Map<String, serde_json::Value>,
}

impl BatchCall {
    /// Fold inline keys into `args`: the flattened form is the second thing
    /// every model tries, and rejecting it costs a whole turn.
    fn normalized(mut self) -> Self {
        if self.inline.is_empty() {
            return self;
        }
        let inline = std::mem::take(&mut self.inline);
        match &mut self.args {
            serde_json::Value::Null => self.args = serde_json::Value::Object(inline),
            serde_json::Value::Object(map) => {
                for (key, value) in inline {
                    // An explicit `args` entry wins over an inline duplicate.
                    map.entry(key).or_insert(value);
                }
            }
            _ => {}
        }
        self
    }
}

/// One `calls[]` entry after lenient parsing.
#[derive(Debug)]
enum ParsedItem {
    Call(BatchCall),
    /// Unparseable — reported as that item's rejection instead of failing the
    /// whole batch, and named from whatever the item did carry.
    Bad {
        name: String,
        reason: String,
    },
}

/// What a host tells `batch_execute` about its own observation tools. The
/// kernel names none: by default the description is generic and a nameless
/// item is rejected rather than guessed.
#[derive(Clone, Default)]
pub struct BatchProfile {
    /// The tool description shown to the model, naming the host's own
    /// observation tools. `None` uses the kernel's generic description.
    pub description: Option<&'static str>,
    /// One example call, as JSON, quoted back in the shape hint.
    pub example_call: Option<&'static str>,
    /// Signature inference for nameless items: if the args carry any of the
    /// keys, the item is that tool. Checked in order; first match wins.
    pub infer: Vec<(&'static [&'static str], &'static str)>,
}

impl BatchProfile {
    fn example_call(&self) -> &str {
        self.example_call
            .unwrap_or(r#"{"tool":"<observation tool>","args":{}}"#)
    }

    /// The documented shape, quoted back at the model when an item makes no sense.
    fn shape_hint(&self) -> String {
        format!(
            "`{{calls: [{{tool, args}}]}}` — e.g. \
             {{\"calls\":[{}]}} \
             (`args` may be omitted and its keys written inline next to `tool`)",
            self.example_call()
        )
    }
}

/// Hoist the function-call shapes models actually emit onto `{tool, args}`
/// before the strict derive runs: the OpenAI wire spelling
/// (`{"function": "x"}` / `{"function": {"name","arguments"}}`), a nested
/// `{"tool": {"name": "x"}}`, and the `call`/`command`/`action` aliases.
/// Without this each becomes a `(unnamed)` Bad item — parseable intent
/// rejected on a label technicality.
fn normalize_shape(mut item: serde_json::Value, profile: &BatchProfile) -> serde_json::Value {
    let Some(obj) = item.as_object_mut() else {
        return item;
    };
    let named = ["tool", "name", "tool_name"]
        .iter()
        .any(|k| obj.get(*k).is_some_and(|v| v.is_string()));
    if named {
        return item;
    }
    // `{"tool": {"name": "x"}}` — the name nested one level down.
    if let Some(n) = obj
        .get("tool")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("name"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
    {
        obj.insert("tool".into(), n.into());
        return item;
    }
    // `{"function": "x"}` or `{"function": {"name", "arguments"}}` —
    // `arguments` folds into `args` when no explicit `args` exists.
    let hoisted: Option<(Option<String>, Option<serde_json::Value>)> = match obj.get("function") {
        Some(serde_json::Value::String(s)) => Some((Some(s.clone()), None)),
        Some(serde_json::Value::Object(f)) => Some((
            f.get("name").and_then(|v| v.as_str()).map(str::to_string),
            f.get("arguments").cloned(),
        )),
        _ => None,
    };
    let args_missing = obj.get("args").is_none();
    if let Some((name, fargs)) = hoisted {
        if let Some(n) = name {
            obj.insert("tool".into(), n.into());
        }
        if args_missing {
            if let Some(a) = fargs {
                obj.insert("args".into(), a);
            }
        }
    }
    for key in ["call", "command", "action"] {
        if obj.get("tool").is_some_and(|v| v.is_string()) {
            break;
        }
        if let Some(s) = obj.get(key).and_then(|v| v.as_str()).map(str::to_string) {
            obj.insert("tool".into(), s.into());
        }
    }
    // No name anywhere — infer from the args signature the host declared,
    // when it's unambiguous. A guessed read-only tool beats a rejected row:
    // worst case it reads the wrong thing once.
    if obj.get("tool").is_none() {
        let args_obj = obj
            .get("args")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_else(|| {
                // Inline args — the item itself is the args object.
                obj.iter()
                    .filter(|(k, _)| {
                        ![
                            "args",
                            "arguments",
                            "type",
                            "call",
                            "command",
                            "action",
                            "function",
                            "name",
                            "tool_name",
                        ]
                        .contains(&k.as_str())
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            });
        let has = |k: &str| args_obj.contains_key(k);
        let inferred = profile
            .infer
            .iter()
            .find(|(keys, _)| keys.iter().any(|k| has(k)))
            .map(|(_, tool)| *tool);
        if let Some(t) = inferred {
            obj.insert("tool".into(), t.into());
            // Inline keys carry the args — make them the args object too so
            // `{"max_hits":…,"path":…}` doesn't reach the tool as `args:null`.
            if obj.get("args").is_none() && obj.get("arguments").is_none() {
                let inline: serde_json::Map<String, serde_json::Value> = obj
                    .iter()
                    .filter(|(k, _)| {
                        ![
                            "tool",
                            "type",
                            "call",
                            "command",
                            "action",
                            "function",
                            "name",
                            "tool_name",
                        ]
                        .contains(&k.as_str())
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                obj.insert("args".into(), serde_json::Value::Object(inline));
            }
        }
    }
    item
}

fn json_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Parse `{calls: […]}` leniently: the strict derive first, and anything it
/// rejects becomes a per-item failure that quotes what arrived. The
/// envelope itself is flexible — models also send a bare array, a single
/// `{"tool","args"}` object, or the array under `items`/`tool_calls`/
/// `batch`/`requests`/`operations`; each unwraps to the same calls list.
fn parse_items(
    args: serde_json::Value,
    profile: &BatchProfile,
) -> Result<Vec<ParsedItem>, ToolError> {
    let raw = if let Some(c) = args.get("calls") {
        c.clone()
    } else if let Some(alt) = ["items", "tool_calls", "batch", "requests", "operations"]
        .iter()
        .find_map(|k| args.get(*k))
    {
        alt.clone()
    } else if let Some(arr) = args.as_array() {
        // A bare `[{tool,args},…]` — the envelope key omitted entirely.
        serde_json::Value::Array(arr.clone())
    } else if args.get("tool").is_some() || args.get("function").is_some() {
        // A single call object — wrap it as a one-item batch.
        serde_json::Value::Array(vec![args])
    } else {
        return Err(ToolError::Args(format!(
            "batch_execute args: missing field `calls`\nExpected {}",
            profile.shape_hint()
        )));
    };
    let items = raw.as_array().cloned().ok_or_else(|| {
        ToolError::Args(format!(
            "batch_execute args: `calls` must be an array, got {}\nExpected {}",
            json_kind(&raw),
            profile.shape_hint()
        ))
    })?;
    Ok(items
        .into_iter()
        .map(|item| {
            match serde_json::from_value::<BatchCall>(normalize_shape(item.clone(), profile)) {
                Ok(call) => ParsedItem::Call(call.normalized()),
                Err(e) => {
                    let name = ["tool", "name", "tool_name"]
                        .iter()
                        .find_map(|k| item.get(k).and_then(|v| v.as_str()))
                        .unwrap_or("(unnamed)")
                        .to_string();
                    let mut shown = item.to_string();
                    if shown.len() > 160 {
                        shown.truncate(shown.floor_char_boundary(160));
                        shown.push('…');
                    }
                    ParsedItem::Bad {
                        name,
                        reason: format!(
                            "item could not be parsed ({e}); got {shown} — expected \
                         {{tool, args}} with `tool` naming an Observation tool"
                        ),
                    }
                }
            }
        })
        .collect())
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct BatchArgs {
    /// Read only by `schema_for` — runtime parsing goes through `parse_items`,
    /// which is lenient about the shapes models actually emit.
    #[allow(dead_code)]

    /// Up to 16 independent calls — results come back in THIS order.
    calls: Vec<BatchCall>,
}

/// Generic description: names no tool, since the kernel ships none.
const GENERIC_DESCRIPTION: &str = "The DEFAULT way to issue 2+ INDEPENDENT observation calls \
     — pack them into ONE round trip instead of firing read-only tools back to \
     back. Args shape is always an object with a `calls` array, and every item \
     MUST carry a `tool` string naming an available observation tool: \
     {\"calls\":[{\"tool\":\"<tool>\",\"args\":{}}]}. Only go sequential when \
     the next call's arguments depend on this call's result. Calls run at \
     limited concurrency (max 4, max 4 network calls); results return in the \
     order you sent them, and one call's failure doesn't fail the batch (check \
     each `── [i] ──` section). Forbidden by design: anything that is not an \
     observation, delegate, todo, ask_question, and nested batch_execute.";

/// The kernel's `batch_execute`, naming no tool.
pub fn spec() -> ToolSpec {
    spec_with(BatchProfile::default())
}

/// `batch_execute` described in terms of the host's own observation tools.
pub fn spec_with(profile: BatchProfile) -> ToolSpec {
    let profile = Arc::new(profile);
    ToolSpec {
        name: "batch_execute",
        schema: super::registry::schema_for::<BatchArgs>(
            profile.description.unwrap_or(GENERIC_DESCRIPTION),
        ),
        readonly: true, // orchestrator — only ever runs auto-allowed Observation specs
        class: ToolClass::Orchestration,
        network: false,
        staged: false,
        review: None,
        cancel_aware: false,
        exec: Arc::new(move |args, ctx| exec(args, ctx, profile.clone()).boxed()),
    }
}

/// Batch eligibility — class is the whole story; no name lists.
fn is_batchable(spec: &ToolSpec) -> bool {
    spec.class == ToolClass::Observation
}

async fn exec(
    args: Args,
    ctx: Arc<ToolCtx>,
    profile: Arc<BatchProfile>,
) -> Result<ToolResult, ToolError> {
    let items = parse_items(args, &profile)?;
    if items.is_empty() {
        return Err(ToolError::Args("`calls` must not be empty".into()));
    }
    if items.len() > MAX_BATCH_CALLS {
        return Err(ToolError::Args(format!(
            "at most {MAX_BATCH_CALLS} calls per batch — split the rest"
        )));
    }

    let registry = ctx
        .active_registry
        .read()
        .unwrap()
        .clone()
        .ok_or_else(|| ToolError::Failed("registry unavailable".into()))?;

    // Network sub-cap — count up front so over-capacity items get a
    // per-item rejection instead of silently queueing.
    let mut network_used = 0usize;
    let mut eligible_flags: Vec<Result<Arc<ToolSpec>, String>> = Vec::with_capacity(items.len());
    for item in &items {
        let ParsedItem::Call(call) = item else {
            let ParsedItem::Bad { reason, .. } = item else {
                unreachable!()
            };
            eligible_flags.push(Err(reason.clone()));
            continue;
        };
        match registry.spec(&call.tool) {
            None => {
                let plugins = ctx.plugins();
                let is_plugin = plugins
                    .as_ref()
                    .and_then(|h| h.read().ok().and_then(|g| g.clone()))
                    .is_some_and(|mgr| {
                        mgr.exported_tools()
                            .iter()
                            .any(|t| t["function"]["name"].as_str() == Some(call.tool.as_str()))
                    });
                eligible_flags.push(Err(if is_plugin {
                    format!(
                        "`{}` is a plugin tool — call it directly; plugin calls can't be batched",
                        call.tool
                    )
                } else {
                    format!("unknown tool `{}` — not in the active registry", call.tool)
                }));
            }
            Some(spec) if !is_batchable(&spec) => eligible_flags.push(Err(format!(
                "`{}` is not batchable — only Observation tools run in a batch",
                call.tool
            ))),
            Some(spec) if spec.network => {
                network_used += 1;
                if network_used > MAX_NETWORK_CALLS {
                    eligible_flags.push(Err("network call limit for this batch (4)".into()));
                } else {
                    eligible_flags.push(Ok(spec));
                }
            }
            Some(spec) => eligible_flags.push(Ok(spec)),
        }
    }

    // Batch-scoped cancellation — a watchdog sets the flag on deadline OR
    // parent cancel; every item's ctx carries it so blocking work aborts
    // for real instead of being abandoned by `timeout`.
    let budget = BatchBudget {
        deadline: Instant::now() + MAX_BATCH_WALL,
        cancel: Arc::new(AtomicBool::new(false)),
    };
    {
        let flag = budget.cancel.clone();
        let parent = ctx.cancel.clone();
        let deadline = budget.deadline;
        tokio::spawn(async move {
            loop {
                if Instant::now() >= deadline
                    || parent.as_ref().is_some_and(|c| c.load(Ordering::Relaxed))
                {
                    flag.store(true, Ordering::Relaxed);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
    }

    let started = Instant::now();
    let streams = items
        .into_iter()
        .zip(eligible_flags)
        .enumerate()
        .map(|(i, (item, flag))| {
            let ctx = ctx.clone();
            let batch_cancel = budget.cancel.clone();
            let deadline = budget.deadline;
            let (tool, call_args) = match item {
                ParsedItem::Call(call) => (call.tool, call.args),
                ParsedItem::Bad { name, .. } => (name, serde_json::Value::Null),
            };
            async move {
                let t0 = Instant::now();
                let out = match flag {
                    Err(reason) => {
                        // Rejected items still surface as child rows — the
                        // live stream should show the same `── [i] ──`
                        // children the result text replays, not silently
                        // drop the ones the batch refused.
                        if let Some(ui) = ctx.ui() {
                            let _ = ui.send(agent_ipc::events::UiEvent::ToolCallStarted {
                                name: tool.clone(),
                                parent: Some("batch_execute".into()),
                                args_preview: String::new(),
                                child_id: None,
                            });
                            let _ = ui.send(agent_ipc::events::UiEvent::ToolCallFinished {
                                name: tool.clone(),
                                ok: false,
                                content: reason.clone(),
                                // `rejected` ui_type — the row renders 已拒绝,
                                // not 已中断: the call never ran.
                                ui_type: Some("rejected".into()),
                                parent: Some("batch_execute".into()),
                                child_id: None,
                            });
                        }
                        ItemOut {
                            tool,
                            status: ItemStatus::Rejected,
                            body: reason,
                            images: Vec::new(),
                            duration_ms: 0,
                        }
                    }
                    Ok(spec) => {
                        // Item ctx carries the batch flag — tools that poll
                        // cancel abort their own blocking work too.
                        let item_ctx = Arc::new(ctx.with_cancel(batch_cancel.clone()));
                        if let Some(ui) = ctx.ui() {
                            let _ = ui.send(agent_ipc::events::UiEvent::ToolCallStarted {
                                name: tool.clone(),
                                // Nested — the UI renders this under the
                                // batch_execute capsule, not top-level.
                                parent: Some("batch_execute".into()),
                                args_preview: call_args
                                    .get("path")
                                    .or_else(|| call_args.get("pattern"))
                                    .or_else(|| call_args.get("url"))
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .chars()
                                    .take(120)
                                    .collect(),
                                child_id: None,
                            });
                        }
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        let run = (spec.exec)(call_args, item_ctx);
                        let out = match tokio::time::timeout(remaining, run).await {
                            Err(_) => ItemOut {
                                tool: tool.clone(),
                                status: if batch_cancel.load(Ordering::Relaxed) {
                                    ItemStatus::Cancelled
                                } else {
                                    ItemStatus::Timeout
                                },
                                body: "batch wall-time budget exhausted".into(),
                                images: Vec::new(),
                                duration_ms: t0.elapsed().as_millis() as u64,
                            },
                            Ok(Ok(res)) => {
                                // Readonly tools never stage writes — a
                                // pending_write here would be dropped by
                                // design anyway (observations don't write).
                                // Images are kept: a batched `screenshot` must
                                // reach the model with the same frame a plain
                                // call would have handed it.
                                ItemOut {
                                    tool: tool.clone(),
                                    status: ItemStatus::Ok,
                                    body: res.content,
                                    images: res.images,
                                    duration_ms: t0.elapsed().as_millis() as u64,
                                }
                            }
                            Ok(Err(e)) => ItemOut {
                                tool: tool.clone(),
                                status: ItemStatus::Error,
                                body: e.to_string(),
                                images: Vec::new(),
                                duration_ms: t0.elapsed().as_millis() as u64,
                            },
                        };
                        if let Some(ui) = ctx.ui() {
                            let (ok, content) = match &out.status {
                                ItemStatus::Ok => (true, out.body.clone()),
                                _ => (false, out.body.clone()),
                            };
                            let _ = ui.send(agent_ipc::events::UiEvent::ToolCallFinished {
                                name: tool,
                                ok,
                                content,
                                ui_type: None,
                                parent: Some("batch_execute".into()),
                                child_id: None,
                            });
                        }
                        out
                    }
                };
                (i, out)
            }
        });

    // `buffered` = bounded concurrency + INPUT order preserved — the model
    // maps `── [i] ──` sections back to its own call list.
    let results: Vec<ItemOut> = futures::stream::iter(streams)
        .buffered(MAX_BATCH_CONCURRENCY)
        .map(|(_, out)| out)
        .collect()
        .await;

    // Sectioned model-facing text — one block per call, order preserved,
    // aggregate bytes capped. Status + duration ride in every header.
    let ok = results
        .iter()
        .filter(|r| matches!(r.status, ItemStatus::Ok))
        .count();
    let errors = results.len() - ok;
    let mut out = format!(
        "[batch_execute — {} calls · {} ok · {} failed · {:.1}s]\n",
        results.len(),
        ok,
        errors,
        started.elapsed().as_secs_f32()
    );
    let mut bytes = out.len();
    let mut capped = false;
    for (i, r) in results.iter().enumerate() {
        let status = match &r.status {
            ItemStatus::Ok => format!("ok · {}ms", r.duration_ms),
            ItemStatus::Error => "error".into(),
            ItemStatus::Timeout => "timeout".into(),
            ItemStatus::Rejected => "rejected".into(),
            ItemStatus::Cancelled => "cancelled".into(),
        };
        let section = if capped {
            format!(
                "\n── [{i}] {} ({status}) ── [output dropped — batch output cap reached]\n",
                r.tool
            )
        } else {
            let room = MAX_BATCH_OUTPUT.saturating_sub(bytes + 96);
            let body = if r.body.len() > room {
                capped = true;
                &r.body[..r.body.floor_char_boundary(room)]
            } else {
                r.body.as_str()
            };
            format!("\n── [{i}] {} ({status}) ──\n{body}\n", r.tool)
        };
        bytes += section.len();
        out.push_str(&section);
    }
    if capped {
        out.push_str("\n⚠ batch output cap reached — later sections were truncated or dropped\n");
    }
    // Item images ride the batch result — the engine attaches them to this
    // tool message, so a batched `screenshot` still reaches the model's eyes.
    // The text sections already name each item's tool, so the model can pair
    // a frame with its call.
    let images: Vec<agent_llm_core::types::ImageRef> =
        results.into_iter().flat_map(|r| r.images).collect();
    let mut res = ToolResult::text(out);
    res.images = images;
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligibility_matrix() {
        let mut spec = spec();
        for (class, want) in [
            (ToolClass::Observation, true),
            (ToolClass::WorkspaceMutation, false),
            (ToolClass::SessionMutation, false),
            (ToolClass::Process, false),
            (ToolClass::Control, false),
            (ToolClass::Orchestration, false),
            (ToolClass::HumanInteraction, false),
        ] {
            spec.class = class;
            assert_eq!(is_batchable(&spec), want, "{class:?}");
        }
    }

    /// Every shape a model reaches for must land as a usable call — the old
    /// strict derive failed the whole batch with a bare `missing field \`tool\``.
    #[test]
    fn item_shapes_the_model_writes_all_parse() {
        let items = parse_items(
            serde_json::json!({"calls": [
                {"tool": "smart_read", "args": {"path": "a.rs"}},
                {"name": "list_dir", "arguments": {"path": "."}},
                {"tool_name": "smart_grep", "pattern": "TODO"},
                {"tool": "smart_read", "path": "b.rs", "start": 3}
            ]}),
            &BatchProfile::default(),
        )
        .unwrap();
        assert_eq!(items.len(), 4);
        for item in &items {
            let ParsedItem::Call(call) = item else {
                panic!("unparsed: {item:?}")
            };
            assert!(!call.tool.is_empty());
            assert!(
                call.args.is_object(),
                "{} args not normalized: {:?}",
                call.tool,
                call.args
            );
        }
        let ParsedItem::Call(grep) = &items[2] else {
            unreachable!()
        };
        assert_eq!(grep.args["pattern"], "TODO", "inline keys must survive");
    }

    #[test]
    fn nameless_items_recover_by_signature() {
        let profile = BatchProfile {
            infer: vec![(&["url"], "web_fetch"), (&["max_hits"], "smart_grep")],
            ..Default::default()
        };
        let items = parse_items(serde_json::json!({"calls": [
            // The real-world miss: args only, no tool key anywhere.
            {"args": {"max_hits": 40, "path": "src"}},
            {"function": "smart_read", "args": {"path": "a.rs"}},
            {"function": {"name": "list_dir", "arguments": {"path": "."}}},
            {"type": "function", "function": {"name": "web_fetch", "arguments": {"url": "https://x"}}},
            {"tool": {"name": "smart_read"}, "args": {"path": "b.rs"}},
            // Genuinely ambiguous — `path` alone could be read or list.
            {"path": "orphan.rs"}
        ]}), &profile)
        .unwrap();
        let name_of = |i: &ParsedItem| -> String {
            match i {
                ParsedItem::Call(c) => c.tool.clone(),
                ParsedItem::Bad { name, .. } => name.clone(),
            }
        };
        assert_eq!(name_of(&items[0]), "smart_grep");
        assert_eq!(name_of(&items[1]), "smart_read");
        assert_eq!(name_of(&items[2]), "list_dir");
        assert_eq!(name_of(&items[3]), "web_fetch");
        assert_eq!(name_of(&items[4]), "smart_read");
        assert_eq!(name_of(&items[5]), "(unnamed)");
        // The hoisted args reach the call, not a null.
        let ParsedItem::Call(c) = &items[3] else {
            unreachable!()
        };
        assert_eq!(c.args["url"], "https://x");
    }

    #[test]
    fn envelope_variants_all_unwrap_to_calls() {
        // Bare array — the `calls` key omitted entirely.
        let items = parse_items(
            serde_json::json!([
                {"tool": "list_dir", "args": {"path": "."}}
            ]),
            &BatchProfile::default(),
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        // Alternate envelope keys.
        for key in ["items", "tool_calls", "batch", "requests", "operations"] {
            let items = parse_items(
                serde_json::json!({key: [{"tool": "list_dir"}]}),
                &BatchProfile::default(),
            )
            .unwrap();
            assert_eq!(items.len(), 1, "{key}");
        }
        // A single call object — the model forgot it was batching.
        let items = parse_items(
            serde_json::json!({"tool": "smart_read", "args": {"path": "a"}}),
            &BatchProfile::default(),
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        // `function` alone still counts as a call-shaped object.
        let items = parse_items(
            serde_json::json!({"function": "smart_read"}),
            &BatchProfile::default(),
        )
        .unwrap();
        assert_eq!(items.len(), 1);
        // Truly unrecognised envelopes still name the shape.
        let e = format!(
            "{}",
            parse_items(serde_json::json!({"nope": 1}), &BatchProfile::default()).unwrap_err()
        );
        assert!(e.contains("missing field `calls`"), "{e}");
    }

    #[test]
    fn a_bad_item_is_rejected_alone_with_the_expected_shape() {
        let items = parse_items(
            serde_json::json!({"calls": [
                {"tool": "list_dir", "args": {"path": "."}},
                {"path": "orphan.rs"},
                7
            ]}),
            &BatchProfile::default(),
        )
        .unwrap();
        assert!(matches!(items[0], ParsedItem::Call(_)));
        let ParsedItem::Bad { name, reason } = &items[1] else {
            panic!("expected a rejected item")
        };
        assert_eq!(name, "(unnamed)");
        assert!(reason.contains("tool"), "{reason}");
        assert!(matches!(items[2], ParsedItem::Bad { .. }));
    }

    /// The kernel ships no tools, so it must not invent a tool name.
    #[test]
    fn the_generic_profile_never_guesses_a_tool_or_names_one() {
        let items = parse_items(
            serde_json::json!({"calls": [{"url": "https://x"}, {"pattern": "TODO", "max_hits": 5}]}),
            &BatchProfile::default(),
        )
        .unwrap();
        assert!(items.iter().all(|i| matches!(i, ParsedItem::Bad { .. })));
        let description = spec().schema["description"].as_str().unwrap().to_string();
        for name in ["smart_read", "smart_grep", "web_fetch", "bash", "list_dir"] {
            assert!(!description.contains(name), "{name} in {description}");
        }
        let hint = BatchProfile::default().shape_hint();
        assert!(!hint.contains("smart_read"), "{hint}");
    }

    #[test]
    fn a_missing_or_non_array_calls_field_names_the_shape() {
        let e = format!(
            "{}",
            parse_items(serde_json::json!({}), &BatchProfile::default()).unwrap_err()
        );
        assert!(e.contains("missing field `calls`"), "{e}");
        assert!(e.contains("{calls:"), "{e}");
        let e = format!(
            "{}",
            parse_items(serde_json::json!({"calls": {}}), &BatchProfile::default()).unwrap_err()
        );
        assert!(e.contains("must be an array"), "{e}");
    }
}
