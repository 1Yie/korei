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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::{Mutex as StdMutex, OnceLock};

use futures::FutureExt;
use serde::{Deserialize, Serialize};

use super::registry::{schema_for, Args, ToolCtx, ToolError, ToolResult, ToolSpec};

/// Cap on the progress note: it rides in the list the model reads back on every
/// call, so an essay about one task would crowd out the tasks themselves.
const NOTE_MAX: usize = 200;

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
///
/// Installed *shared*: a delegated child writes the same board its parent does,
/// so a child that finishes a step marks that step done instead of keeping a
/// private list nobody reads. Several items may be in progress at once — one per
/// agent working — and the store's own lock serialises their writes.
pub struct TodoStorageSlot(pub Arc<dyn TodoStorage>);

/// Install a list a delegated child shares with its parent.
pub fn install_shared(ctx: &mut ToolCtx, storage: Arc<dyn TodoStorage>) {
    ctx.ext.insert_shared(Arc::new(TodoStorageSlot(storage)));
}

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

/// Where one item stands. Rows written before this existed carry `done: bool`
/// instead and are read through [`StoredItem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl Default for TodoStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// One todo entry. A child names its parent; nesting stops at one level.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "StoredItem")]
pub struct TodoItem {
    pub id: usize,
    pub text: String,
    pub status: TodoStatus,
    pub parent: Option<usize>,
    /// How the item is being carried out, in one line: what the model writes when
    /// it starts the task. Only meaningful while it is in progress, so `done`
    /// drops it rather than leaving a stale note on every finished row.
    pub note: Option<String>,
}

/// The row as it is read: `status`/`parent` for what this version writes,
/// `done` for what the previous one wrote, all optional so either row loads.
/// Only reading goes through here — a save writes [`TodoItem`]'s own shape, so
/// the legacy field disappears the first time the list is touched.
#[derive(Debug, Deserialize)]
struct StoredItem {
    id: usize,
    text: String,
    #[serde(default)]
    status: Option<TodoStatus>,
    #[serde(default)]
    parent: Option<usize>,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    done: Option<bool>,
}

impl From<StoredItem> for TodoItem {
    fn from(s: StoredItem) -> Self {
        let status = s.status.unwrap_or(if s.done.unwrap_or(false) {
            TodoStatus::Completed
        } else {
            TodoStatus::Pending
        });
        Self {
            id: s.id,
            text: s.text,
            status,
            parent: s.parent,
            note: s.note,
        }
    }
}

/// Persisted store — the `state/<ws>/<id>/todos.json` row.
#[derive(Debug, Default, Serialize, Deserialize)]
struct TodoStore {
    next_id: usize,
    items: Vec<TodoItem>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct TodoArgs {
    /// What to do: `add` (needs `text`, optional `parent`), `list`,
    /// `start`/`done`/`undone`/`remove` (need `id`), `clear` (wipes the list).
    action: String,
    /// Todo text — required for `add`.
    text: Option<String>,
    /// Item to act on — required for `start`/`done`/`undone`/`remove`. Either the
    /// number or the label the list printed (`2`, `1-3`).
    id: Option<TodoRef>,
    /// Parent id — `add` only: nests the new item under that one. One level;
    /// a parent that is itself a child is refused.
    parent: Option<usize>,
    /// `start` only: one line on how the item is being carried out, shown beside
    /// it. Sending `start` again replaces it; an empty string clears it. Dropped
    /// when the item is marked `done`.
    note: Option<String>,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: "todo",
        schema: schema_for::<TodoArgs>(
            "Manage this session's task list (persisted with the session).\n\
             Actions: `add` a task (`text`; `parent` nests it under an existing\n\
             task), `list` all tasks, `start` it (`id`, plus `note`: one line on\n\
             how it is being done) when work on it begins, `done`/`undone`\n\
             (`id`), `remove` (`id`, takes its children with it), `clear` (wipe\n\
             the list). One level of nesting, and a task with an open child\n\
             cannot be `done`. Several tasks may be in progress at once (one per\n\
             agent working, a delegated child included), so `start` does not\n\
             clear the others; call it again to keep a task's `note` current —\n\
             that note is what the user reads while the task runs. Use it to\n\
             track multi-step work the user can see — plan, subtasks (as\n\
             children), follow-ups.",
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

/// Serialize complete board updates across contexts sharing a store row.
/// Loading and saving under separate locks can lose concurrent edits.
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

/// One item's mark in the rendered block.
fn mark(status: TodoStatus) -> &'static str {
    match status {
        TodoStatus::Pending => "[ ]",
        TodoStatus::InProgress => "[~]",
        TodoStatus::Completed => "[x]",
    }
}

/// Items in display order with their depth: roots by id, each followed by its
/// own children by id. A child whose parent is gone (a hand-edited row) is
/// shown as a root instead of vanishing from the list.
fn ordered(items: &[TodoItem]) -> Vec<(&TodoItem, usize)> {
    let ids: HashSet<usize> = items.iter().map(|t| t.id).collect();
    let mut roots: Vec<&TodoItem> = items
        .iter()
        .filter(|t| t.parent.map_or(true, |p| !ids.contains(&p)))
        .collect();
    roots.sort_by_key(|t| t.id);
    let mut out = Vec::with_capacity(items.len());
    for root in roots {
        out.push((root, 0));
        let mut children: Vec<&TodoItem> =
            items.iter().filter(|c| c.parent == Some(root.id)).collect();
        children.sort_by_key(|c| c.id);
        out.extend(children.into_iter().map(|c| (c, 1)));
    }
    out
}

/// How a caller names an item: the number it was given, or the hierarchical
/// label the list shows (`1`, `1-2`). The label is what the model sees, so it is
/// what it will pass back.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum TodoRef {
    Id(usize),
    Label(String),
}

/// A child's place in its parent's list, 1-based, in id order.
fn child_position(items: &[TodoItem], id: usize) -> Option<usize> {
    let parent = items.iter().find(|t| t.id == id)?.parent?;
    let mut siblings: Vec<usize> = items
        .iter()
        .filter(|t| t.parent == Some(parent))
        .map(|t| t.id)
        .collect();
    siblings.sort_unstable();
    siblings.iter().position(|sid| *sid == id).map(|p| p + 1)
}

/// What a caller wrote, for an error message.
fn want_label(want: &TodoRef) -> String {
    match want {
        TodoRef::Id(n) => format!("#{n}"),
        TodoRef::Label(raw) => format!("#{}", raw.trim()),
    }
}

/// The label every item prints under: roots numbered by where they sit in the
/// list (`1`, `2`, …) and children by their place under that root (`2-1`). Not
/// the stored id — three subtasks under the first task must not push the second
/// task to `#4`. What the list prints is what a call names, so `resolve` reads
/// the same numbering.
fn labels(items: &[TodoItem]) -> HashMap<usize, String> {
    let mut out = HashMap::new();
    let mut root_no = 0usize;
    for (t, depth) in ordered(items) {
        if depth == 0 {
            root_no += 1;
            out.insert(t.id, root_no.to_string());
        } else {
            let pos = child_position(items, t.id).unwrap_or(0);
            out.insert(t.id, format!("{root_no}-{pos}"));
        }
    }
    out
}

/// Resolve what a caller named to the stored id. The printed label wins; a
/// stored id still resolves when no label reads the same, so a call written
/// against an older list lands somewhere sensible instead of failing.
fn resolve(items: &[TodoItem], want: &TodoRef) -> Option<usize> {
    let by_label = labels(items);
    let find_label = |text: &str| {
        by_label
            .iter()
            .find(|(_, label)| label.as_str() == text)
            .map(|(id, _)| *id)
    };
    match want {
        TodoRef::Id(n) => {
            find_label(&n.to_string()).or_else(|| items.iter().find(|t| t.id == *n).map(|t| t.id))
        }
        TodoRef::Label(raw) => {
            let raw = raw.trim();
            find_label(raw).or_else(|| {
                raw.parse::<usize>()
                    .ok()
                    .and_then(|n| items.iter().find(|t| t.id == n).map(|t| t.id))
            })
        }
    }
}

/// Render the list as a compact `[ ]/[~]/[x] #label text` block for the model,
/// children indented one level under their parent and numbered `parent-child`.
fn render(items: &[TodoItem]) -> String {
    if items.is_empty() {
        return "No todos.".into();
    }
    let done = items
        .iter()
        .filter(|t| t.status == TodoStatus::Completed)
        .count();
    let labels = labels(items);
    let mut s = format!("{}/{} done\n", done, items.len());
    for (t, depth) in ordered(items) {
        let label = labels
            .get(&t.id)
            .cloned()
            .unwrap_or_else(|| t.id.to_string());
        let note = t
            .note
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(|n| format!(" — {n}"))
            .unwrap_or_default();
        s.push_str(&format!(
            "{}{} #{} {}{note}\n",
            "  ".repeat(depth),
            mark(t.status),
            label,
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
            // One level only: the parent must exist and be a root itself.
            let parent = match a.parent {
                Some(p) => {
                    let parent = resolve(&store.items, &TodoRef::Id(p))
                        .and_then(|id| store.items.iter().find(|t| t.id == id))
                        .ok_or_else(|| {
                            ToolError::Failed(format!(
                                "parent #{p} not found — run `list` for live ids"
                            ))
                        })?;
                    if let Some(grand) = parent.parent {
                        return Err(ToolError::Failed(format!(
                            "todo #{p} is a child of #{grand} — todos nest one level"
                        )));
                    }
                    Some(parent.id)
                }
                None => None,
            };
            let item = TodoItem {
                id: store.next_id,
                text: text.into(),
                status: TodoStatus::Pending,
                parent,
                note: None,
            };
            store.next_id += 1;
            store.items.push(item.clone());
            save(sess.as_ref(), &store)?;
            let label = labels(&store.items)
                .get(&item.id)
                .cloned()
                .unwrap_or_else(|| item.id.to_string());
            format!("Added #{label} {}\n\n{}", item.text, render(&store.items))
        }
        "start" | "done" | "undone" => {
            let want =
                a.id.as_ref()
                    .ok_or_else(|| ToolError::Args(format!("`{}` needs an `id`", a.action)))?;
            let id = resolve(&store.items, want).ok_or_else(|| {
                ToolError::Failed(format!(
                    "todo {} not found — run `list` for live ids",
                    want_label(want)
                ))
            })?;
            // A completed parent over an open child reads as a plan that lies,
            // so the child has to finish (or go) first.
            if a.action == "done" {
                let open = store
                    .items
                    .iter()
                    .find(|t| t.parent == Some(id) && t.status != TodoStatus::Completed);
                if let Some(child) = open {
                    return Err(ToolError::Failed(format!(
                        "todo #{id} still has open child #{} — finish or remove it first",
                        child.id
                    )));
                }
            }
            let item = store.items.iter_mut().find(|t| t.id == id).ok_or_else(|| {
                ToolError::Failed(format!("todo #{id} not found — run `list` for live ids"))
            })?;
            item.status = match a.action.as_str() {
                "done" => TodoStatus::Completed,
                "start" => TodoStatus::InProgress,
                _ => TodoStatus::Pending,
            };
            // The note describes work in progress. `start` without one leaves the
            // current note alone (a second `start` is also how it is updated); an
            // empty one clears it; a finished or reopened item carries none.
            match a.action.as_str() {
                "start" => {
                    if let Some(raw) = a.note.as_deref() {
                        let trimmed = raw.trim();
                        item.note = if trimmed.is_empty() {
                            None
                        } else {
                            Some(trimmed.chars().take(NOTE_MAX).collect())
                        };
                    }
                }
                _ => item.note = None,
            }
            let word = match item.status {
                TodoStatus::Completed => "done",
                TodoStatus::InProgress => "in progress",
                TodoStatus::Pending => "reopened",
            };
            save(sess.as_ref(), &store)?;
            let label = labels(&store.items)
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.to_string());
            format!("#{label} {word}\n\n{}", render(&store.items))
        }
        "remove" => {
            let want =
                a.id.as_ref()
                    .ok_or_else(|| ToolError::Args("`remove` needs an `id`".into()))?;
            let id = resolve(&store.items, want).ok_or_else(|| {
                ToolError::Failed(format!(
                    "todo {} not found — run `list` for live ids",
                    want_label(want)
                ))
            })?;
            // A parent's children go with it: an orphan would render as a root
            // and read as a task nobody planned.
            let gone = labels(&store.items)
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.to_string());
            let before = store.items.len();
            store.items.retain(|t| t.id != id && t.parent != Some(id));
            let removed = before - store.items.len();
            if removed == 0 {
                return Err(ToolError::Failed(format!(
                    "todo #{id} not found — run `list` for live ids"
                )));
            }
            save(sess.as_ref(), &store)?;
            let lead = if removed > 1 {
                format!("Removed #{gone} and {} children", removed - 1)
            } else {
                format!("Removed #{gone}")
            };
            format!("{lead}\n\n{}", render(&store.items))
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
                "unknown action '{other}' — use add|list|start|done|undone|remove|clear"
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
        let _ = exec(serde_json::json!({"action":"add","text":"c"}), ctx.clone())
            .await
            .unwrap();
        // The printed label is positional, so a recycled stored id would be
        // invisible in the block — read the row that was actually written.
        let store = load(storage(&ctx).as_ref()).await.unwrap();
        let fresh = store.items.iter().find(|t| t.text == "c").unwrap();
        assert_eq!(fresh.id, 3, "id recycled after clear");
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
        let store = load(storage(&ctx).as_ref()).await.unwrap();
        let next = store.items.iter().find(|t| t.text == "new").unwrap();
        assert_eq!(next.id, 11, "duplicate id handed out: {}", r.content);
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
        // Numbering is positional: one task is #1 whatever id it was stored under.
        assert!(r.content.contains("#1 old task"), "{}", r.content);
        // The new row is durable, THEN the host drops its source.
        assert!(storage.row.lock().unwrap().is_some());
        assert!(
            storage.seeded.load(std::sync::atomic::Ordering::SeqCst),
            "migration should let the host clean its copy"
        );
        // …and the new home holds it (ids included).
        exec(
            serde_json::json!({"action":"add","text":"next"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        // The stored id keeps climbing from the adopted row; the printed label
        // is positional, so the id is checked on the row itself.
        let store = load(storage.as_ref()).await.unwrap();
        let next = store.items.iter().find(|t| t.text == "next").unwrap();
        assert_eq!(next.id, 4, "stored id did not follow the adopted row");
    }

    /// A delegated child must write the SAME board: parallel children each
    /// finish their own task, and the parent has to see it.
    #[test]
    fn a_child_context_inherits_the_shared_list() {
        let mut parent = ToolCtx::new();
        install_shared(&mut parent, storage_with(None));
        let child = parent.ext.inheritable();
        assert!(
            child.get::<TodoStorageSlot>().is_some(),
            "a child context did not inherit the session's todo storage"
        );
    }

    /// Several items may be in progress at once — one per agent working.
    #[tokio::test]
    async fn two_tasks_can_be_in_progress_together() {
        let ctx = ctx();
        for text in ["alpha", "beta"] {
            exec(serde_json::json!({"action":"add","text":text}), ctx.clone())
                .await
                .unwrap();
        }
        exec(
            serde_json::json!({"action":"start","id":1,"note":"子代理 A"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let r = exec(
            serde_json::json!({"action":"start","id":2,"note":"子代理 B"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(
            r.content.contains("[~] #1 alpha — 子代理 A"),
            "{}",
            r.content
        );
        assert!(
            r.content.contains("[~] #2 beta — 子代理 B"),
            "{}",
            r.content
        );
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

    /// A row written before `status` existed must still load: `done` decides.
    /// The first write after that drops the old field for good.
    #[tokio::test]
    async fn legacy_rows_load_and_are_rewritten_in_the_new_shape() {
        let store = storage_with(Some(
            r#"{"next_id":3,"items":[{"id":1,"text":"old open","done":false},{"id":2,"text":"old done","done":true}]}"#,
        ));
        let ctx = ctx_over(store.clone());
        let r = exec(serde_json::json!({"action":"list"}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("[ ] #1 old open"), "{}", r.content);
        assert!(r.content.contains("[x] #2 old done"), "{}", r.content);

        exec(
            serde_json::json!({"action":"add","text":"new"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let row = String::from_utf8(store.row.lock().unwrap().clone().unwrap()).unwrap();
        assert!(
            !row.contains("\"done\""),
            "the legacy field survived a write: {row}"
        );
        assert!(row.contains("\"status\""), "{row}");
    }

    /// The label the list prints is what the caller may pass back.
    #[tokio::test]
    async fn a_hierarchical_label_addresses_the_child() {
        let ctx = ctx();
        for (text, parent) in [("parent", None), ("first", Some(1)), ("second", Some(1))] {
            let args = match parent {
                Some(p) => serde_json::json!({"action":"add","text":text,"parent":p}),
                None => serde_json::json!({"action":"add","text":text}),
            };
            exec(args, ctx.clone()).await.unwrap();
        }
        let r = exec(
            serde_json::json!({"action":"start","id":"1-2","note":"在跑"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(
            r.content.contains("[~] #1-2 second — 在跑"),
            "{}",
            r.content
        );
        // The first child keeps its label through the change.
        let r = exec(serde_json::json!({"action":"done","id":"1-1"}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("[x] #1-1 first"), "{}", r.content);

        let err = exec(serde_json::json!({"action":"done","id":"1-9"}), ctx.clone())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("#1-9"), "{err}");
    }

    /// A subtask must not push the next task's number: three children under `#1`
    /// leave the next task at `#2`, not `#5`.
    #[tokio::test]
    async fn a_subtask_does_not_shift_the_next_task_number() {
        let ctx = ctx();
        exec(
            serde_json::json!({"action":"add","text":"第一项"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        for text in ["子一", "子二"] {
            exec(
                serde_json::json!({"action":"add","text":text,"parent":1}),
                ctx.clone(),
            )
            .await
            .unwrap();
        }
        let r = exec(
            serde_json::json!({"action":"add","text":"第三项"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(
            r.content.ends_with(
                "0/4 done\n[ ] #1 第一项\n  [ ] #1-1 子一\n  [ ] #1-2 子二\n[ ] #2 第三项"
            ),
            "{}",
            r.content
        );

        let r = exec(
            serde_json::json!({"action":"add","text":"next child","parent":2}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(r.content.contains("  [ ] #2-1 next child"), "{}", r.content);

        // And the printed label is what addresses it.
        let r = exec(
            serde_json::json!({"action":"start","id":"1-2","note":"在跑"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(r.content.contains("[~] #1-2 子二 — 在跑"), "{}", r.content);
        let r = exec(serde_json::json!({"action":"done","id":"1-2"}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("[x] #1-2 子二"), "{}", r.content);
    }

    /// A child is indented under its parent, and the parent keeps its own row.
    #[tokio::test]
    async fn add_with_parent_nests_one_level() {
        let ctx = ctx();
        exec(
            serde_json::json!({"action":"add","text":"parent"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let r = exec(
            serde_json::json!({"action":"add","text":"child","parent":1}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(r.content.contains("[ ] #1 parent"), "{}", r.content);
        // A child is numbered under its parent, not by its stored id.
        assert!(r.content.contains("  [ ] #1-1 child"), "{}", r.content);
    }

    #[tokio::test]
    async fn a_grandchild_is_refused() {
        let ctx = ctx();
        exec(
            serde_json::json!({"action":"add","text":"root"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        exec(
            serde_json::json!({"action":"add","text":"child","parent":1}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let err = exec(
            serde_json::json!({"action":"add","text":"grandchild","parent":2}),
            ctx.clone(),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("nest one level"), "{err}");
    }

    /// A completed parent over an open child would read as a plan that lies.
    #[tokio::test]
    async fn done_waits_for_open_children() {
        let ctx = ctx();
        exec(
            serde_json::json!({"action":"add","text":"parent"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        exec(
            serde_json::json!({"action":"add","text":"child","parent":1}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let err = exec(serde_json::json!({"action":"done","id":1}), ctx.clone())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("open child #2"), "{err}");

        exec(serde_json::json!({"action":"done","id":2}), ctx.clone())
            .await
            .unwrap();
        let r = exec(serde_json::json!({"action":"done","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("2/2 done"), "{}", r.content);
    }

    #[tokio::test]
    async fn removing_a_parent_takes_its_children() {
        let ctx = ctx();
        exec(
            serde_json::json!({"action":"add","text":"parent"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        exec(
            serde_json::json!({"action":"add","text":"child","parent":1}),
            ctx.clone(),
        )
        .await
        .unwrap();
        exec(
            serde_json::json!({"action":"add","text":"other"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let r = exec(serde_json::json!({"action":"remove","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(
            r.content.contains("Removed #1 and 1 children"),
            "{}",
            r.content
        );
        assert!(!r.content.contains("#1-1"), "{}", r.content);
        // Renumbered by position: the one task left is #1.
        assert!(r.content.contains("[ ] #1 other"), "{}", r.content);
    }

    /// The note is the "how it is going" line: written by `start`, updated by a
    /// second `start`, cleared by an empty one and dropped when the item ends.
    #[tokio::test]
    async fn start_writes_and_clears_the_progress_note() {
        let ctx = ctx();
        exec(
            serde_json::json!({"action":"add","text":"换掉解析器"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let r = exec(
            serde_json::json!({"action":"start","id":1,"note":"先读 parser.rs"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(
            r.content.contains("[~] #1 换掉解析器 — 先读 parser.rs"),
            "{}",
            r.content
        );

        // A second start keeps the note unless it carries a new one.
        let r = exec(serde_json::json!({"action":"start","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("— 先读 parser.rs"), "{}", r.content);

        let r = exec(
            serde_json::json!({"action":"start","id":1,"note":"改完解析器，正在补测试"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(
            r.content.contains("— 改完解析器，正在补测试"),
            "{}",
            r.content
        );
        assert!(!r.content.contains("先读 parser.rs"), "{}", r.content);

        // An empty note clears it.
        let r = exec(
            serde_json::json!({"action":"start","id":1,"note":"  "}),
            ctx.clone(),
        )
        .await
        .unwrap();
        assert!(!r.content.contains("—"), "{}", r.content);

        // Done drops it: the line described work in progress.
        exec(
            serde_json::json!({"action":"start","id":1,"note":"收尾"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let r = exec(serde_json::json!({"action":"done","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("[x] #1 换掉解析器"), "{}", r.content);
        assert!(!r.content.contains("— 收尾"), "{}", r.content);
    }

    /// The note rides the block the model reads on every call, so it is capped.
    #[tokio::test]
    async fn a_long_note_is_cut_to_its_cap() {
        let ctx = ctx();
        exec(serde_json::json!({"action":"add","text":"t"}), ctx.clone())
            .await
            .unwrap();
        let long = "说".repeat(NOTE_MAX + 50);
        let r = exec(
            serde_json::json!({"action":"start","id":1,"note":long}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let line = r.content.lines().find(|l| l.starts_with("[~]")).unwrap();
        assert_eq!(line.chars().filter(|c| *c == '说').count(), NOTE_MAX);
    }

    #[tokio::test]
    async fn start_marks_in_progress_and_undone_reopens() {
        let ctx = ctx();
        exec(
            serde_json::json!({"action":"add","text":"task"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let r = exec(serde_json::json!({"action":"start","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("#1 in progress"), "{}", r.content);
        assert!(r.content.contains("[~] #1 task"), "{}", r.content);

        let r = exec(serde_json::json!({"action":"undone","id":1}), ctx.clone())
            .await
            .unwrap();
        assert!(r.content.contains("#1 reopened"), "{}", r.content);
        assert!(r.content.contains("[ ] #1 task"), "{}", r.content);
    }

    /// Children render directly under their parent, in id order, one level in.
    #[tokio::test]
    async fn nested_block_renders_children_under_their_parent() {
        let ctx = ctx();
        exec(serde_json::json!({"action":"add","text":"A"}), ctx.clone())
            .await
            .unwrap();
        exec(
            serde_json::json!({"action":"add","text":"A child","parent":1}),
            ctx.clone(),
        )
        .await
        .unwrap();
        exec(serde_json::json!({"action":"add","text":"B"}), ctx.clone())
            .await
            .unwrap();
        exec(
            serde_json::json!({"action":"start","id":"1-1"}),
            ctx.clone(),
        )
        .await
        .unwrap();
        let r = exec(serde_json::json!({"action":"list"}), ctx.clone())
            .await
            .unwrap();
        assert_eq!(
            r.content,
            "0/3 done\n[ ] #1 A\n  [~] #1-1 A child\n[ ] #2 B"
        );
    }
}
