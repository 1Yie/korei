//! The kernel's extension seams are its semver surface. This file implements
//! every seam from outside the crate, using only the public API, so a change
//! to a seam's shape (a new required method, a removed field) breaks the build
//! here before it breaks a product. Adding a defaulted method must not.

use std::sync::Arc;

use agent_kernel::plugins::{PluginHandle, PluginRouter};
use agent_kernel::tools::batch::BatchProfile;
use agent_kernel::tools::delegate::{ChildSpec, SubagentHost};
use agent_kernel::tools::registry::{ToolCtx, ToolResult};
use agent_kernel::tools::todo::{TodoStorage, TodoStorageSlot};
use agent_kernel::turn_policy::{Effect, NoopPolicy, TurnPolicy, TurnView, Verdict};
use agent_llm_core::ChatMessage;
use serde_json::Value;

struct Policy;

#[async_trait::async_trait]
impl TurnPolicy for Policy {
    fn begin_turn(&self) {}
    async fn before_dispatch(&self, _tool: &str, _is_readonly: bool) -> Option<String> {
        None
    }
    async fn on_quiet_round(&self, view: &dyn TurnView) -> Verdict {
        if view.did_tool_work() && view.mode() == "x" {
            return Verdict::Nudge {
                hidden: "go on".into(),
            };
        }
        Verdict::Finish {
            before_final: vec![Effect {
                history: None,
                ui: None,
            }],
            after_final: Vec::new(),
        }
    }
}

struct Router;

#[async_trait::async_trait]
impl PluginRouter for Router {
    fn exported_tools(&self) -> Vec<Value> {
        Vec::new()
    }
    async fn dispatch_tool_call(
        &self,
        _name: &str,
        _args: Value,
    ) -> Option<Result<ToolResult, String>> {
        None
    }
}

/// Only the required methods: a defaulted addition must keep this compiling.
struct Host;
impl SubagentHost for Host {}

struct Storage;
impl TodoStorage for Storage {
    fn read(&self) -> std::io::Result<Option<Vec<u8>>> {
        Ok(None)
    }
    fn write(&self, _bytes: &[u8]) -> std::io::Result<()> {
        Ok(())
    }
    fn key(&self) -> String {
        "k".into()
    }
}

#[test]
fn every_seam_is_implementable_from_outside() {
    let _: Arc<dyn TurnPolicy> = Arc::new(Policy);
    let _: Arc<dyn TurnPolicy> = Arc::new(NoopPolicy);
    let _: Arc<dyn PluginRouter> = Arc::new(Router);
    let _handle: PluginHandle = Default::default();
    let host: Arc<dyn SubagentHost> = Arc::new(Host);
    assert!(host.resolve(&ToolCtx::new(), "nobody").is_none());
    assert!(host
        .instructions(&ToolCtx::new(), &ChildSpec::default())
        .is_empty());
    let mut ctx = ToolCtx::new();
    ctx.ext.insert(Arc::new(TodoStorageSlot(Arc::new(Storage))));
    let _ = BatchProfile::default();
    let _ = ChatMessage::user("x");
}
