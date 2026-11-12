//! Code Mode's model control surface and turn-borrowed host execution contract.

use agent_llm_core::types::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::{
    engine::{Engine, EngineIo},
    invocation::{InvocationStopReason, InvocationTurn},
};

pub const CONTROL_TOOL_NAME: &str = "execute_code";
pub const MAX_MODEL_RESULT_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngineStrategy {
    #[default]
    ReAct,
    CodeMode,
}

/// Untrusted program inputs. Identity and effective limits come from the host.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeModeRequest {
    pub code: String,
    pub params: Value,
    pub limits: Option<Map<String, Value>>,
}

impl CodeModeRequest {
    pub fn parse(arguments: &str) -> Result<Self, String> {
        let value = crate::schema::parse_arguments(arguments).map_err(String::from)?;
        crate::schema::validate(&control_schema()["function"]["parameters"], &value)
            .map_err(String::from)?;
        serde_json::from_value(value).map_err(|error| error.to_string())
    }
}

pub fn control_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": CONTROL_TOOL_NAME,
            "description": "Run a JavaScript async function body to orchestrate authorized tools.\nDiscover names with tools.list({cursor: null}), inspect schemas with\ntools.describe(name), and invoke tools with await tools.call(name, args).\nUse params for input data, console for bounded diagnostics, and return\nthe JSON result needed by the model. Files, network, environment, and\nmodule loading are available only through authorized tools.\nA human refusal ends this program and returns control to the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "JavaScript async function body."},
                    "params": {"description": "JSON input data available through the params binding."},
                    "limits": {
                        "type": "object",
                        "description": "Optional requested execution limits, clamped to host limits."
                    }
                },
                "required": ["code", "params"],
                "additionalProperties": false
            }
        }
    })
}

/// Resources borrowed from the current turn, including its shared policy gates.
pub struct CodeModeContext<'turn> {
    pub engine: &'turn Engine,
    pub io: &'turn EngineIo,
    pub turn: &'turn InvocationTurn,
    pub control_call: &'turn ToolCall,
    pub tool_ctx: &'turn std::sync::Arc<crate::tools::registry::ToolCtx>,
    pub parent_identity: Option<&'turn crate::invocation_scope::InvocationIdentity>,
    pub parent_deadline: Option<std::time::Instant>,
    /// Kernel-generated outer request identity, independent of guest input.
    pub request_id: u64,
    /// Host must reserve an attempt from this allowance before each inner
    /// dispatch, including concurrent dispatches. Creating a scope cannot reset it.
    /// usize::MAX means no configured parent call cap. Every program still
    /// requires independent finite host runtime and dispatch limits.
    pub remaining_tool_calls: usize,
}

/// Only this explicit model projection reaches outer history and tool events.
pub struct CodeModeOutcome {
    pub model_result: String,
    pub is_error: bool,
    /// Actual gateway attempts, including failed attempts; never a guest count.
    pub attempted_calls: usize,
    /// Trusted host provenance. Human refusal must come from
    /// `InvocationAudit.stop_reason`, never denied flags or error text.
    pub stop_reason: Option<InvocationStopReason>,
}

impl CodeModeOutcome {
    pub(crate) fn rejected(message: impl Into<String>) -> Self {
        Self {
            model_result: message.into(),
            is_error: true,
            attempted_calls: 0,
            stop_reason: None,
        }
    }

    pub(crate) fn bound_projection(&mut self) {
        if self.model_result.len() > MAX_MODEL_RESULT_BYTES {
            const SUFFIX: &str = "\n[model result truncated]";
            let mut end = MAX_MODEL_RESULT_BYTES - SUFFIX.len();
            while !self.model_result.is_char_boundary(end) {
                end -= 1;
            }
            self.model_result.truncate(end);
            self.model_result.push_str(SUFFIX);
        }
    }
}

/// The host owns the VM, effective limits, execution identity, scope, artifacts
/// and audit. It must stop admission, drain gateway completions, finalize
/// artifacts and terminal audit, then emit one execution terminal and return.
/// Cancellation is cooperative: the kernel awaits cleanup instead of dropping
/// this future. All capabilities must use the context's turn and scoped gateway.
#[async_trait::async_trait]
pub trait CodeModeHandler: Send + Sync {
    async fn execute(
        &self,
        request: CodeModeRequest,
        context: CodeModeContext<'_>,
    ) -> CodeModeOutcome;
}
