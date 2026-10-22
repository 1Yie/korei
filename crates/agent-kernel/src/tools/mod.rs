//! Generic conversation tools and their registry.

pub mod ask;
pub mod link;
pub mod batch;
pub mod builtins;
pub mod delegate;
pub mod host_slots;
pub mod registry;
pub mod todo;

pub use registry::{Args, ToolContext, ToolCtx, ToolRegistry, ToolResult, ToolSpec};
