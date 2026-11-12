//! agent-kernel — pure async orchestration engine.
//!
//! Dependency law (workspace-layout.md): no iced, no reqwest. Kernel sees
//! only normalized `StreamChunk` from agent-llm and typed events to/from
//! agent-ipc (Stage 4).
//!
//! Stage 3 scope: tool registry + phase-1 built-ins.
//! Stage 4 scope: `AgentState` machine + `SessionActor` + `channels` — the
//! ReAct loop runs headless on a scripted provider (`cargo test -p agent-kernel`).

pub mod channels;
pub mod code_mode;
pub mod compaction;
pub mod engine;
pub mod hooks;
pub mod invocation_scope;
pub mod permissions;
pub mod plugins;
pub(crate) use agent_tool::schema;
pub mod tools;
pub mod turn_policy;
pub use engine::invocation;
