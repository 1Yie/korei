//! Host-owned execution services and lifetime, independent of engine authority.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use crate::{
    engine::PluginHandle, hooks::HookChain, invocation::AuditSink, tools::registry::ToolCtx,
};

/// Identity supplied by trusted Rust host code, never tool arguments.
#[derive(Clone, Debug)]
pub struct InvocationIdentity {
    pub execution_id: String,
    pub agent_id: String,
    pub session_id: String,
    pub trace_id: Option<String>,
    pub parent_call_id: Option<String>,
}

/// Owner of one execution. Call `close` before releasing its host resources.
pub struct InvocationScope {
    pub(crate) ctx: Arc<ToolCtx>,
    pub(crate) binding: Arc<ScopeBinding>,
}

pub(crate) struct ScopeBinding {
    pub identity: InvocationIdentity,
    pub hooks: HookChain,
    pub plugins: Option<PluginHandle>,
    pub audit: Option<Arc<dyn AuditSink>>,
    pub deadline: Option<Instant>,
    pub lifetime: Arc<ExecutionLifetime>,
}

/// Services built for a child's fresh ToolCtx by the trusted host.
pub struct ScopedChildServices {
    pub hooks: HookChain,
    pub plugins: Option<PluginHandle>,
    pub strategy: crate::code_mode::EngineStrategy,
    pub code_mode_handler: Option<Arc<dyn crate::code_mode::CodeModeHandler>>,
    pub max_code_mode_tool_calls: Option<usize>,
}

pub type ScopedChildFactory =
    dyn Fn(&ToolCtx, &mut ToolCtx) -> Result<ScopedChildServices, String> + Send + Sync;

pub(crate) struct InvocationParent(pub String);

impl ScopeBinding {
    pub fn check_open(&self) -> Result<(), String> {
        if self
            .deadline
            .is_some_and(|d| tokio::time::Instant::now().into_std() >= d)
        {
            return Err("invocation scope deadline exceeded".into());
        }
        self.lifetime.check_open()
    }

    pub async fn cancelled(&self) {
        loop {
            if self.check_open().is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    pub fn child(
        &self,
        agent_id: String,
        parent_call_id: String,
        services: ScopedChildServices,
    ) -> Arc<Self> {
        let mut identity = self.identity.clone();
        identity.agent_id = agent_id;
        identity.parent_call_id = Some(parent_call_id);
        Arc::new(Self {
            identity,
            hooks: services.hooks,
            plugins: services.plugins,
            audit: self.audit.clone(),
            deadline: self.deadline,
            lifetime: self.lifetime.clone(),
        })
    }
}

impl InvocationScope {
    pub fn new(
        ctx: Arc<ToolCtx>,
        identity: InvocationIdentity,
        hooks: HookChain,
        plugins: Option<PluginHandle>,
        audit: Option<Arc<dyn AuditSink>>,
        deadline: Option<Instant>,
    ) -> Self {
        let binding = Arc::new(ScopeBinding {
            identity,
            hooks,
            plugins,
            audit,
            deadline,
            lifetime: Arc::new(ExecutionLifetime::default()),
        });
        let mut ctx = ctx.with_cancel(binding.lifetime.cancel.clone());
        ctx.ext.insert(binding.clone());
        Self {
            ctx: Arc::new(ctx),
            binding,
        }
    }

    pub fn ctx(&self) -> &Arc<ToolCtx> {
        &self.ctx
    }

    pub fn cancel(&self) {
        self.binding.lifetime.cancel.store(true, Ordering::Release);
    }

    /// Reject new work, cancel existing work and await every invocation/task.
    /// A timeout retains ownership: the host must retain resources and retry.
    pub async fn close(&self) -> Result<(), String> {
        self.binding.lifetime.close();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let changed = self.binding.lifetime.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if self
                    .binding
                    .lifetime
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .active
                    == 0
                {
                    return;
                }
                changed.await;
            }
        })
        .await
        .map_err(|_| "invocation scope cleanup timed out; resources remain owned".into())
    }
}

impl Drop for InvocationScope {
    fn drop(&mut self) {
        self.binding.lifetime.close();
    }
}

#[derive(Default)]
struct LifetimeState {
    closed: bool,
    active: usize,
    children: std::collections::HashSet<u64>,
}

#[derive(Default)]
pub(crate) struct ExecutionLifetime {
    pub cancel: Arc<AtomicBool>,
    state: Mutex<LifetimeState>,
    changed: tokio::sync::Notify,
}

pub(crate) struct ExecutionLease(Arc<ExecutionLifetime>);

impl ExecutionLifetime {
    pub fn register_child(&self, id: u64) {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .children
            .insert(id);
    }

    pub fn owns_child(&self, id: u64) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .children
            .contains(&id)
    }

    pub fn retain(self: &Arc<Self>) -> ExecutionLease {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).active += 1;
        ExecutionLease(self.clone())
    }

    pub fn check_open(&self) -> Result<(), String> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed || self.cancel.load(Ordering::Acquire) {
            Err("invocation scope is closed".into())
        } else {
            Ok(())
        }
    }

    pub fn enter(self: &Arc<Self>) -> Result<ExecutionLease, String> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed || self.cancel.load(Ordering::Acquire) {
            return Err("invocation scope is closed".into());
        }
        state.active += 1;
        Ok(ExecutionLease(self.clone()))
    }

    pub fn close(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
        self.cancel.store(true, Ordering::Release);
        self.changed.notify_waiters();
    }
}

impl Drop for ExecutionLease {
    fn drop(&mut self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active -= 1;
        self.0.changed.notify_waiters();
    }
}

/// Whether UI state mirroring must leave hook dispatch to the scoped owner.
pub fn is_invocation_scoped(ctx: &ToolCtx) -> bool {
    ctx.ext.get::<ScopeBinding>().is_some()
}
