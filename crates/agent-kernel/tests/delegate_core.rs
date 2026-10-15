use std::sync::Arc;

use agent_kernel::tools::delegate::{ChildRoute, SubagentSpawner};
use agent_kernel::tools::{ToolCtx, ToolRegistry};
use agent_testkit::ScriptedProvider;

#[tokio::test]
async fn default_spawner_delegates_without_workspace_or_host_profiles() {
    let provider = Arc::new(ScriptedProvider::new());
    provider.script_text("CHILD-RESULT");
    let registry = Arc::new(ToolRegistry::child_builtins());
    let route = Arc::new(std::sync::RwLock::new(ChildRoute {
        provider,
        model: "test".into(),
        temperature: 0.0,
        context_window: 256_000,
        thinking_level: None,
        thinking_level_map: None,
        model_params: Default::default(),
        model_input: Vec::new(),
        models: Default::default(),
    }));
    let mut ctx = ToolCtx::new();
    ctx.install_subagent(SubagentSpawner::new(route, registry.clone(), registry));
    let ctx = Arc::new(ctx);
    let spec = agent_kernel::tools::delegate::spec();
    let err = (spec.exec)(
        serde_json::json!({"task": "inspect", "agent": "missing"}),
        ctx.clone(),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("host catalog"), "{err}");
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        (spec.exec)(serde_json::json!({"task": "inspect"}), ctx),
    )
    .await
    .expect("scripted child must finish")
    .expect("unnamed delegation needs no host adapter");
    assert!(out.content.contains("CHILD-RESULT"), "{}", out.content);
}
