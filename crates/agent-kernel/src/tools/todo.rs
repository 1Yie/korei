//! `todo` — session-scoped task list the agent maintains while it works.
//!
//! Where the list lives is the host's choice: it installs a [`TodoStorage`] on
//! the context. The coding host keeps it beside the session's history (a new
//! session starts clean, a resumed one keeps its list, and the user's repo
//! stays free of agent scratch files); a context with none gets an in-memory
//! list that lives as long as the context. Either way `readonly: true` is
//! honest — the tool never touches a workspace, so it needs no approval.
//!
//! Two invariants: one load→mutate→save per store row at a time
//! (`state_lock`), and ids are never reused, not even across `clear` — a
//! recycled `#1` makes the model mark the wrong item done.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{Mutex as StdMutex, OnceLock};

use futures::FutureExt;
use serde::{Deserialize, Serialize};

use super::registry::{schema_for, Args, ToolCtx, ToolError, ToolResult, ToolSpec};

/// Where one session's todo list is persisted: a single opaque row. The tool
/// owns the row's shape (JSON) and its locking; the host owns where the bytes
/// go. `key` names the list — the host picks it per context so two sessions
/// never share a row.
pub trait TodoStorage: Send + Sync {
    /// The stored row, `None` when the list was never written.
    fn read(&self) -> std::io::Result<Option<Vec<u8>>>;
    /// Replace the row atomically — a crash must show the previous row or the
    /// new one, never a torn one.
    fn write(&self, bytes: &[u8]) -> std::io::Result<()>;
    /// Stable identity of this list, used to serialise load→mutate→save.
    fn key(&self) -> String;
    /// A one-time seed for a list that has never been written (a host's own
    /// legacy file, say). `Ok(None)` = nothing to migrate. Called only while
    /// the row is absent; returning bytes does NOT write them — the tool does,
    /// and only then calls [`TodoStorage::seeded`] so the host can drop its
    /// legacy copy once the new one is durable.
    fn seed(&self) -> std::io::Result<Option<Vec<u8>>> {
        Ok(None)
    }
    /// The seed is durable in the new row; the host may remove its source.
    fn seeded(&self) {}
}

/// The host's installed storage for this context's todo list.
pub struct TodoStorageSlot(pub Arc<dyn TodoStorage>);

/// The default when a host installed nothing: the list lives in memory for as
/// long as the context does. Nothing else about the tool changes.
#[derive(Default)]
pub struct MemoryTodos {
    row: StdMutex<Option<Vec<u8>>>,
}

impl TodoStorage for MemoryTodos {
    fn read(&self) -> std::io::Result<Option<Vec<u8>>> {
        Ok(self.row.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }
    fn write(&self, bytes: &[u8]) -> std::io::Result<()> {
        *self.row.lock().unwrap_or_else(|e| e.into_inner()) = Some(bytes.to_vec());
        Ok(())
    }
    fn key(&self) -> String {
        // Per context, never shared: the address of this very row.
        format!("mem:{:p}", self)
    }
}

/// One todo entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: usize,
    pub text: String,
    pub done: bool,
}

/// Persisted store — the `state/<ws>/<id>/todos.json` row.
#[derive(Debug, Default, Serialize, Deserialize)]
struct TodoStore {
    next_id: usize,
    items: Vec<TodoItem>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct TodoArgs {
    /// What to do: `add` (needs `text`), `list`, `done`/`undone`/`remove`
    /// (need `id`), `clear` (wipes the list).
    action: String,
    /// Todo text — required for `add`.
    text: Option<String>,
    /// Todo id — required for `done`/`undone`/`remove`.
    id: Option<usize>,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: "todo",
        schema: schema_for::<TodoArgs>(
            "Manage this session's task list (persisted with the session).\n\
             Actions: `add` a task (`text`), `list` all tasks, `done`/`undone`\n\
             (`id`), `remove` (`id`), `clear` (wipe the list). Use it to track\n\
             multi-step work the user can see — plan, subtasks, follow-ups.",
        ),
        readonly: true,
        class: super::registry::ToolClass::SessionMutation,
        network: false,
        staged: false,
        review: None,
        cancel_aware: false,
        exec: std::sync::Arc::new(|args, ctx| exec(args, ctx).boxed()),
    }
}

/// The context's storage: the host's slot, else the in-memory list the
/// context itself carries (`ToolCtx::todo_fallback`), shared by every call on
/// it and dropped with it.
fn storage(ctx: &ToolCtx) -> Arc<dyn TodoStorage> {
    if let Some(slot) = ctx.ext.get::<TodoStorageSlot>() {
        return slot.0.clone();
    }
    ctx.todo_fallback.clone()
}

/// One async lock per store row, shared process-wide.
///
/// `exec` is a read-modify-write cycle, so two overlapping invocations can
/// lose an update or hand the same id to two items. Today the engine
/// dispatches a round's calls sequentially — this is insurance, not a fix for
/// a reproducible bug — but the failure is *silent* (a todo just vanishes),
/// and parallel dispatch is the direction the runtime is heading. Keyed by
/// store key rather than held in `ToolCtx` so it also covers a second
/// `ToolCtx` over the same session (subagent, headless context).
fn state_lock(key: String) -> Arc<tokio::sync::Mutex<()>> {
    static LOCKS: OnceLock<StdMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> =
        OnceLock::new();
    let locks = LOCKS.get_or_init(|| StdMutex::new(HashMap::new()));
    let mut locks = locks.lock().unwrap_or_else(|e| e.into_inner());
    locks.entry(key).or_default().clone()
}

async fn load(store: &dyn TodoStorage) -> Result<TodoStore, ToolError> {
    match store.read() {
        Ok(Some(bytes)) => serde_json::from_slice(&bytes)
            .map_err(|e| ToolError::Failed(format!("corrupt todo list row: {e}"))),
        Ok(None) => match store.seed() {
            // The host has a legacy copy to adopt. Unreadable seed state is NOT
            // an empty list: treating it as one would strand the user's tasks
            // and the host drops its copy a line later. Surface it, leave it.
            Ok(Some(bytes)) => {
                let seeded: TodoStore = serde_json::from_slice(&bytes).map_err(|e| {
                    ToolError::Failed(format!(
                        "legacy todo state is not valid todo JSON: {e} — left in place; \
                         fix or delete it and retry"
                    ))
                })?;
                // Copy first, let the host drop its source only after the new
                // copy is durable — a failed migration must not be destructive.
                save(store, &seeded)?;
                store.seeded();
                Ok(seeded)
            }
            Ok(None) => Ok(TodoStore {
                next_id: 1,
                items: Vec::new(),
            }),
            Err(e) => Err(ToolError::Io(e)),
        },
        Err(e) => Err(ToolError::Io(e)),
    }
}

/// Single-row put — the host's write IS the atomic replace, so a crash shows
/// either the previous row or the new one, never a truncated file.
fn save(store: &dyn TodoStorage, todos: &TodoStore) -> Result<(), ToolError> {
    let json = serde_json::to_vec_pretty(todos)
        .map_err(|e| ToolError::Failed(format!("serialize todos: {e}")))?;
    store.write(&json).map_err(ToolError::Io)
}

/// Render the list as a compact `[ ]/[x] #id text` block for the model.
fn render(items: &[TodoItem]) -> String {
    if items.is_empty() {
        return "No todos.".into();
    }
    let done = items.iter().filter(|t| t.done).count();
    let mut s = format!("{}/{} done\n", done, items.len());
    for t in items {
        s.push_str(&format!(
            "[{}] #{} {}\n",
            if t.done { "x" } else { " " },
            t.id,
            t.text
        ));
    }
    s.trim_end().to_string()
}

async fn exec(args: Args, ctx: Arc<ToolCtx>) -> Result<ToolResult, ToolError> {
    let a: TodoArgs =
        serde_json::from_value(args).map_err(|e| ToolError::Args(format!("todo args: {e}")))?;
    let sess = storage(&ctx);
    let lock = state_lock(sess.key());
    let _guard = lock.lock().await;
    let mut store = load(sess.as_ref()).await?;
    // `next_id` must clear the highest stored id — a file with `next_id: 2`
    // sitting next to `#10` (hand-edited, partially migrated, older format)
    // would otherwise hand out a duplicate id.
    let max_id = store.items.iter().map(|t| t.id).max().unwrap_or(0);
    store.next_id = store.next_id.max(max_id + 1);

    let out = match a.action.as_str() {
        "list" => render(&store.items),
        "add" => {
            let text = a
                .text
                .as_deref()
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .ok_or_else(|| ToolError::Args("`add` needs a non-empty `text`".into()))?;
            let item = TodoItem {
                id: store.next_id,
                text: text.into(),
                done: false,
            };
            store.next_id += 1;
            store.items.push(item.clone());
            save(sess.as_ref(), &store)?;
            format!(
                "Added #{} {}\n\n{}",
                item.id,
                item.text,
                render(&store.items)
            )
        }
        "done" | "undone" => {
            let id =
                a.id.ok_or_else(|| ToolError::Args(format!("`{}` needs an `id`", a.action)))?;
            let item = store.items.iter_mut().find(|t| t.id == id).ok_or_else(|| {
                ToolError::Failed(format!("todo #{id} not found — run `list` for live ids"))
            })?;
            item.done = a.action == "done";
            let done = item.done;
            save(sess.as_ref(), &store)?;
            format!(
                "#{} {}\n\n{}",
                id,
                if done { "done" } else { "reopened" },
                render(&store.items)
            )
        }
        "remove" => {
            let id =
                a.id.ok_or_else(|| ToolError::Args("`remove` needs an `id`".into()))?;
            let before = store.items.len();
            store.items.retain(|t| t.id != id);
            if store.items.len() == before {
                return Err(ToolError::Failed(format!(
                    "todo #{id} not found — run `list` for live ids"
                )));
            }
            save(sess.as_ref(), &store)?;
            format!("Removed #{id}\n\n{}", render(&store.items))
        }
        "clear" => {
            let n = store.items.len();
            store.items.clear();
            // `next_id` deliberately survives a clear — see the module docs.
            save(sess.as_ref(), &store)?;
            format!("Cleared {n} todos.")
        }
        other => {
            return Err(ToolError::Args(format!(
                "unknown action '{other}' — use add|list|done|undone|remove|clear"
            )))
        }
    };

    Ok(ToolResult {
        content: out,
        ui_type: Some("todo"),
        fuzzy: false,
        commit: None,
        images: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host's storage for the tests: one row, plus an optional legacy seed
    /// the way the coding host adopts a workspace's old `todos.json`.
    struct TestStorage {
        key: String,
        row: StdMutex<Option<Vec<u8>>>,
        legacy: StdMutex<Option<Vec<u8>>>,
        seeded: std::sync::atomic::AtomicBool,
    }

    impl TodoStorage for TestStorage {
        fn read(&self) -> std::io::Result<Option<Vec<u8>>> {
            Ok(self.row.lock().unwrap().clone())
        }
        fn write(&self, bytes: &[u8]) -> std::io::Result<()> {
            *self.row.lock().unwrap() = Some(bytes.to_vec());
            Ok(())
        }
        fn key(&self) -> String {
            self.key.clone()
        }
        fn seed(&self) -> std::io::Result<Option<Vec<u8>>> {
            Ok(self.legacy.lock().unwrap().clone())
        }
        fn seeded(&self) {
            self.seeded.store(true, std::sync::atomic::Ordering::SeqCst);
            *self.legacy.lock().unwrap() = None;
        }
    }

    fn storage_with(legacy: Option<&str>) -> Arc<TestStorage> {
        // A unique key per test — the lock table is process-wide.
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Arc::new(TestStorage {
            key: format!(
                "test-{}",
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ),
            row: StdMutex::new(None),
            legacy: StdMutex::new(legacy.map(|s| s.as_bytes().to_vec())),
            seeded: std::sync::atomic::AtomicBool::new(false),
        })
    }

    fn ctx_over(storage: Arc<TestStorage>) -> Arc<ToolCtx> {
        let mut ctx = ToolCtx::new();
        ctx.ext.insert(Arc::new(TodoStorageSlot(storage)));
        Arc::new(ctx)
    }

    fn ctx() -> Arc<ToolCtx> {
        ctx_over(storage_with(None))
    }

    #[tokio::test]
    async fn add_list_done_remove_roundtrip() {
        let ctx = ctx();
        let r = exec(
            serde_json::json!({"action":"add","text":"fix the bug"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(r.content.contains("#1"));

        let r = exec(serde_json::json!({"action":"list"}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("fix the bug"));

        let r = exec(serde_json::json!({"action":"done","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("[x]"));

        let r = exec(serde_json::json!({"action":"remove","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("Removed #1"));

        // persisted across "sessions" (fresh load from the same store)
        let store = load(storage(&ctx).as_ref()).await.unwrap();
        assert!(store.items.is_empty());
    }

    /// Ids never recycle — a fresh `#1` for a different task aliases with the
    /// `#1` still in the model's context window (hence: monotonic across
    /// `clear`, and derived from the highest stored id on load).
    #[tokio::test]
    async fn ids_stay_monotonic_across_clear() {
        let ctx = ctx();
        exec(serde_json::json!({"action":"add","text":"a"}), ctx.clone())
            .await
            .unwrap();
        exec(serde_json::json!({"action":"add","text":"b"}), ctx.clone())
            .await
            .unwrap();
        let r = exec(serde_json::json!({"action":"clear"}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("Cleared 2"));
        let r = exec(serde_json::json!({"action":"add","text":"c"}), ctx.clone())
            .await
            .unwrap();
        assert!(
            r.content.contains("#3"),
            "id recycled after clear: {}",
            r.content
        );
    }

    /// A hand-edited / older file can carry a `next_id` below the highest id.
    #[tokio::test]
    async fn next_id_clears_the_highest_stored_id() {
        let ctx = ctx();
        let seeded = serde_json::json!({
            "next_id": 2,
            "items": [{"id": 10, "text": "old task", "done": false}],
        });
        storage(&ctx)
            .write(serde_json::to_string(&seeded).unwrap().as_bytes())
            .unwrap();

        let r = exec(
            serde_json::json!({"action":"add","text":"new"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(
            r.content.contains("#11"),
            "duplicate id handed out: {}",
            r.content
        );
    }

    /// Overlapping invocations must not lose an add. Without the per-store
    /// lock this drops updates: every call would load the same snapshot and
    /// the last save would win.
    #[tokio::test]
    async fn concurrent_adds_neither_collide_nor_vanish() {
        let ctx = ctx();
        let mut set = tokio::task::JoinSet::new();
        for i in 0..8 {
            let ctx = ctx.clone();
            set.spawn(async move {
                exec(
                    serde_json::json!({"action":"add","text":format!("t{i}")}),
                    ctx,
                )
                .await
                .unwrap()
            });
        }
        while set.join_next().await.is_some() {}

        let store = load(storage(&ctx).as_ref()).await.unwrap();
        assert_eq!(store.items.len(), 8, "lost update: {:?}", store.items);
        let mut ids: Vec<usize> = store.items.iter().map(|t| t.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, (1..=8).collect::<Vec<_>>(), "id collision: {ids:?}");
    }

    /// Corrupt legacy state must not be mistaken for "no todos" — that both
    /// strands the user's tasks and (before this) deleted the only copy.
    #[tokio::test]
    async fn corrupt_legacy_migration_fails_loudly_and_keeps_the_source() {
        let storage = storage_with(Some("{ this is not json"));
        let ctx = ctx_over(storage.clone());

        let err = exec(serde_json::json!({"action":"list"}), ctx.clone())
            .await
            .expect_err("a corrupt legacy seed must surface, not read as empty");
        assert!(
            matches!(err, ToolError::Failed(ref m) if m.contains("legacy")),
            "{err}"
        );
        assert!(
            !storage.seeded.load(std::sync::atomic::Ordering::SeqCst),
            "the host was told to drop its copy of an unreadable list"
        );
        assert!(storage.legacy.lock().unwrap().is_some());
        assert!(storage.row.lock().unwrap().is_none(), "nothing was written");
    }

    #[tokio::test]
    async fn legacy_migration_copies_before_removing() {
        let seeded = serde_json::json!({
            "next_id": 4,
            "items": [{"id": 3, "text": "old task", "done": true}],
        });
        let storage = storage_with(Some(&serde_json::to_string(&seeded).unwrap()));
        let ctx = ctx_over(storage.clone());

        let r = exec(serde_json::json!({"action":"list"}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("#3 old task"), "{}", r.content);
        // The new row is durable, THEN the host drops its source.
        assert!(storage.row.lock().unwrap().is_some());
        assert!(
            storage.seeded.load(std::sync::atomic::Ordering::SeqCst),
            "migration should let the host clean its copy"
        );
        // …and the new home holds it (ids included).
        let r = exec(
            serde_json::json!({"action":"add","text":"next"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(r.content.contains("#4 next"), "{}", r.content);
    }

    /// A host that installs nothing still gets a working list: it lives in
    /// memory for the life of the context and is private to it.
    #[tokio::test]
    async fn without_a_host_storage_the_list_lives_in_memory_per_context() {
        let a = Arc::new(ToolCtx::new());
        let b = Arc::new(ToolCtx::new());
        exec(
            serde_json::json!({"action":"add","text":"only in a"}),
            a.clone(),
        )
        .await
        .unwrap();
        let in_a = exec(serde_json::json!({"action":"list"}), a.clone())
            .await
            .unwrap();
        assert!(in_a.content.contains("only in a"), "{}", in_a.content);
        let in_b = exec(serde_json::json!({"action":"list"}), b.clone())
            .await
            .unwrap();
        assert!(
            in_b.content.contains("No todos"),
            "leaked across contexts: {}",
            in_b.content
        );
    }

    /// `save` is one atomic commit — the row parses back and re-saving works.
    #[tokio::test]
    async fn save_commits_one_row_and_roundtrips() {
        let ctx = ctx();

        exec(serde_json::json!({"action":"add","text":"a"}), ctx.clone())
            .await
            .unwrap();

        let raw = storage(&ctx).read().unwrap().unwrap();
        let store: TodoStore = serde_json::from_slice(&raw).unwrap();
        assert_eq!(store.items.len(), 1);
    }

    #[tokio::test]
    async fn ids_survive_restart() {
        let ctx = ctx();
        exec(serde_json::json!({"action":"add","text":"a"}), ctx.clone())
            .await
            .unwrap();
        exec(serde_json::json!({"action":"add","text":"b"}), ctx.clone())
            .await
            .unwrap();
        // Fresh load — next_id must keep climbing, not reset to 1.
        let r = exec(serde_json::json!({"action":"add","text":"c"}), ctx.clone())
            .await
            .unwrap();
        assert!(
            r.content.contains("#3"),
            "ids collided after reload: {}",
            r.content
        );
    }
}
