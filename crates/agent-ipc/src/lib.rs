//! agent-ipc — the channel contract between kernel and every frontend.
//!
//! `UiEvent` lives here, not in the kernel: `agent-kernel` publishes it, the
//! frontends translate, and no frontend imports kernel types for event shapes.
//! Commands are each product's own; the kernel's command channel is generic. `AgentEvent` — the kernel-internal sampler→session
//! feedback — stays in `agent-kernel/src/channels.rs`; it never crosses a
//! transport.

/// Wire codec (length-prefixed JSON frames) — no in-tree consumer yet: the
/// Tauri bridge passes typed values in-process. Kept as the framing contract
/// for out-of-process/stdio frontends.
pub mod codec;
pub mod events;

pub use events::{AgentState, UiEvent};
