//! Nested requests are serviced by the borrowing gateway, not a static executor.

use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

use agent_llm_core::types::ToolCall;
use tokio::sync::{mpsc, oneshot};

use crate::invocation::InvocationOutcome;

pub(crate) const NESTED_QUEUE: usize = 16;
pub(crate) const NESTED_CONCURRENCY: usize = 4;

#[derive(Clone)]
pub struct NestedDispatcher {
    pub(crate) sender: mpsc::Sender<NestedRequest>,
}

pub(crate) struct NestedRequest {
    pub call: ToolCall,
    pub cancel: Arc<AtomicBool>,
    pub deadline: Instant,
    pub rejection: Option<String>,
    pub reply: oneshot::Sender<InvocationOutcome>,
}

impl NestedDispatcher {
    /// Send an observation request to the currently executing parent gateway.
    pub async fn invoke(
        &self,
        call: ToolCall,
        cancel: Arc<AtomicBool>,
        deadline: Instant,
        rejection: Option<String>,
    ) -> Result<InvocationOutcome, String> {
        if call.arguments.len() > agent_tool::schema::MAX_ARGUMENT_BYTES
            || call.name.len() > 1024
            || call.id.len() > 1024
            || rejection.as_ref().is_some_and(|s| s.len() > 16 * 1024)
        {
            return Err("nested invocation exceeds request limits".into());
        }
        let (reply, result) = oneshot::channel();
        self.sender
            .send(NestedRequest {
                call,
                cancel,
                deadline,
                rejection,
                reply,
            })
            .await
            .map_err(|_| "nested invocation gateway is closed".to_string())?;
        result
            .await
            .map_err(|_| "nested invocation parent ended".to_string())
    }
}
