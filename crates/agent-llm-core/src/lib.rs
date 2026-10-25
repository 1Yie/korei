//! agent-llm-core — the provider *contract*.
//!
//! Everything the kernel and tools need to talk about an LLM: normalized
//! `ChatMessage`/`StreamChunk`, the `LlmProvider` trait, `ProviderCompat`
//! dialect flags, the `Sampler` retry/timeout driver and egress masking.
//! Vendor SDKs, HTTP, config files and the credential store live in `agent-llm`.

pub mod compat;
pub mod masking;
pub mod provider;
pub mod sampler;
pub mod thinking;
pub mod types;

pub use compat::ProviderCompat;
pub use masking::EgressMasker;
pub use provider::{BoxStream, Capabilities, LlmProvider, ModelParams, UnavailableProvider};
pub use thinking::{available_thinking_levels, normalize_thinking_level, THINKING_LEVELS};
pub use types::{ChatMessage, ProviderState, Role, StreamChunk, ToolCall};
