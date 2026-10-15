//! Optional external tool routing supplied by a host.

use std::sync::{Arc, RwLock};

use agent_tool::ToolResult;
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait PluginRouter: Send + Sync {
    fn exported_tools(&self) -> Vec<Value>;

    /// `None` means this router does not own the name. An owned call keeps
    /// its success or failure, so a real error is never replaced by fallback.
    async fn dispatch_tool_call(
        &self,
        name: &str,
        args: Value,
    ) -> Option<Result<ToolResult, String>>;
}

pub type PluginHandle = Arc<RwLock<Option<Arc<dyn PluginRouter>>>>;
