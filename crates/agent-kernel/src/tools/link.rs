//! `link_workspace` — "associate that directory with this session".
//!
//! The refusal of a file tool already offers a grant, but a plain instruction
//! ("link `~/Workspace/agent-rs` for me") produces no refusal to hang it on.
//! This is the explicit request: name a directory, and the host decides what a
//! grant means and whether it may be given (the kernel has no idea what a
//! directory permission is — it only carries the question).
//!
//! `Review.risk = RequiresApproval`, so the gate shows the card in every mode
//! that can ask; a headless child has no host slot and refuses outright.

use futures::FutureExt;

use super::registry::{schema_for, Args, Review, ToolCtx, ToolError, ToolResult, ToolSpec};
use agent_tool::CallRisk;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct LinkArgs {
    /// Directory to associate — absolute, or the project root a user named.
    path: String,
    /// `read` (look at it) or `readwrite` (edit it too). Default `readwrite`.
    #[serde(default)]
    access: Option<String>,
    /// `true` keeps the grant for every session of this workspace; false (the
    /// default) scopes it to this conversation.
    #[serde(default)]
    persist: Option<bool>,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: "link_workspace",
        schema: schema_for::<LinkArgs>(
            "Associate a directory outside the workspace with this session, so the file tools \
             and `bash` treat it as in scope (absolute paths only). Use this when the user asks \
             for it, or when a task needs another repository in the same project and the user \
             agrees. The call pauses for approval like any other tool: the user's click IS the \
             grant. `persist: true` keeps it for every session of this workspace; otherwise it \
             ends with this conversation.",
        ),
        readonly: false,
        // Needs a human to consent, and a delegated child has none.
        class: super::registry::ToolClass::HumanInteraction,
        network: false,
        staged: false,
        review: Some(std::sync::Arc::new(review)),
        cancel_aware: false,
        exec: std::sync::Arc::new(|args, ctx| exec(args, ctx).boxed()),
    }
}

fn review(args: &Args, _: &std::sync::Arc<ToolCtx>) -> Review {
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
    let access = args
        .get("access")
        .and_then(|v| v.as_str())
        .filter(|a| *a == "read")
        .unwrap_or("readwrite");
    let scope = if args.get("persist").and_then(|v| v.as_bool()) == Some(true) {
        "本工作区"
    } else {
        "本会话"
    };
    Review {
        // Always asks: widening what a session may touch is never implied by
        // the permission mode.
        risk: CallRisk::RequiresApproval,
        summary: Some(format!("关联 {path}（{access}，{scope}）")),
        ..Default::default()
    }
}

async fn exec(args: Args, ctx: std::sync::Arc<ToolCtx>) -> Result<ToolResult, ToolError> {
    let parsed: LinkArgs =
        serde_json::from_value(args).map_err(|e| ToolError::Args(format!("link_workspace: {e}")))?;
    let path = parsed.path.trim();
    if path.is_empty() {
        return Err(ToolError::Args("`path` must name a directory".into()));
    }
    let access = match parsed.access.as_deref() {
        None | Some("readwrite") => "readwrite",
        // Anything else reads as the narrower grant: an unknown label must
        // never widen access.
        Some(_) => "read",
    };
    let (action, scope) = if parsed.persist == Some(true) {
        ("link:workspace", "本工作区")
    } else {
        ("link:session", "本会话")
    };
    if !ctx.has_scope_grant() {
        return Err(ToolError::Failed(
            "this host cannot link directories — there is nothing installed to grant with".into(),
        ));
    }
    match ctx.grant_scope(action, path, access) {
        Some(Ok(())) => Ok(ToolResult::text(format!(
            "linked {path} ({access}, {scope}) — the file tools and `bash` reach it now. Use its \
             absolute path: relative paths still resolve against the workspace."
        ))),
        Some(Err(msg)) => Err(ToolError::Failed(msg)),
        None => Err(ToolError::Failed("this host cannot link directories".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::registry::ToolCtx;
    use std::sync::Arc;

    fn ctx_with(grant: Option<Arc<dyn Fn(&str, &str, &str) -> Result<(), String> + Send + Sync>>) -> Arc<ToolCtx> {
        let mut ctx = ToolCtx::new();
        if let Some(g) = grant {
            ctx.install_scope_grant(g);
        }
        Arc::new(ctx)
    }

    /// Without a host slot there is nothing to grant with: the tool says so
    /// instead of pretending the directory is now reachable.
    #[tokio::test]
    async fn refuses_when_the_host_cannot_grant() {
        let ctx = ctx_with(None);
        let err = exec(
            serde_json::json!({"path": "/tmp/x"}),
            ctx,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("cannot link directories"), "{err}");
    }

    /// The action and the access reach the host verbatim — that is the whole
    /// contract on this side.
    #[tokio::test]
    async fn passes_the_action_access_and_path_to_the_host() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::<(String, String, String)>::new()));
        let sink = seen.clone();
        let ctx = ctx_with(Some(Arc::new(move |action: &str, path: &str, access: &str| {
            sink.lock()
                .unwrap()
                .push((action.into(), path.into(), access.into()));
            Ok(())
        })));
        let out = exec(
            serde_json::json!({"path": "/w/kernel", "access": "read", "persist": true}),
            ctx,
        )
        .await
        .expect("granted");
        assert!(out.content.contains("/w/kernel"), "{}", out.content);
        assert_eq!(
            *seen.lock().unwrap(),
            [(
                "link:workspace".to_string(),
                "/w/kernel".to_string(),
                "read".to_string()
            )]
        );
    }

    /// An unknown access label is the narrower grant.
    #[tokio::test]
    async fn an_unknown_access_label_reads_as_read() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::<(String, String, String)>::new()));
        let sink = seen.clone();
        let ctx = ctx_with(Some(Arc::new(move |action: &str, path: &str, access: &str| {
            sink.lock()
                .unwrap()
                .push((action.into(), path.into(), access.into()));
            Ok(())
        })));
        exec(serde_json::json!({"path": "/w", "access": "rw-everything"}), ctx)
            .await
            .unwrap();
        assert_eq!(seen.lock().unwrap()[0].2, "read");
    }
}
