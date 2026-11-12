//! Bounded host-authored Code Mode events carried by `UiEvent::Custom`.

use serde::{Deserialize, Serialize};

use crate::UiEvent;

pub const MAX_CODE_MODE_EVENT_BYTES: usize = 16 * 1024;
pub const MAX_CODE_MODE_TEXT_BYTES: usize = 4096;

/// Correlation supplied by the trusted host, never copied from program params.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeModeIdentity {
    pub execution_id: String,
    pub call_id: String,
    pub control_request_id: u64,
    pub parent_call_id: Option<String>,
    pub session_id: String,
    pub agent_id: String,
    pub trace_id: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CodeModePhase {
    Started,
    Progress,
    Finished,
}

impl CodeModePhase {
    pub fn kind(self) -> &'static str {
        match self {
            Self::Started => "code_mode.started",
            Self::Progress => "code_mode.progress",
            Self::Finished => "code_mode.finished",
        }
    }
}

/// Emit Started before progress. Finished is unique and follows completion
/// drain, artifact finalization and terminal audit, even on cancellation.
/// Summary/console contain bounded diagnostics or opaque artifact references;
/// raw capability outputs belong in host-owned artifacts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeModeEvent {
    pub identity: CodeModeIdentity,
    pub phase: CodeModePhase,
    /// Optional inner bridge/approval request identity.
    pub request_id: Option<u64>,
    pub summary: String,
    pub console: Option<String>,
    /// Present only on Finished. False includes partial failures.
    pub ok: Option<bool>,
}

impl CodeModeEvent {
    pub fn into_ui_event(self) -> Result<UiEvent, &'static str> {
        if self.summary.len() > MAX_CODE_MODE_TEXT_BYTES
            || self
                .console
                .as_ref()
                .is_some_and(|text| text.len() > MAX_CODE_MODE_TEXT_BYTES)
        {
            return Err("Code Mode event text exceeds limit");
        }
        if (self.phase == CodeModePhase::Finished) != self.ok.is_some() {
            return Err("Code Mode terminal status does not match phase");
        }
        let identity = &self.identity;
        for value in [
            Some(identity.execution_id.as_str()),
            Some(identity.call_id.as_str()),
            identity.parent_call_id.as_deref(),
            Some(identity.session_id.as_str()),
            Some(identity.agent_id.as_str()),
            identity.trace_id.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if value.is_empty() || value.len() > 1024 {
                return Err("invalid Code Mode event identity");
            }
        }
        let kind = self.phase.kind().to_string();
        let payload = serde_json::to_value(self).map_err(|_| "invalid Code Mode event")?;
        if serde_json::to_vec(&payload)
            .map_err(|_| "invalid Code Mode event")?
            .len()
            > MAX_CODE_MODE_EVENT_BYTES
        {
            return Err("Code Mode event exceeds byte limit");
        }
        Ok(UiEvent::Custom { kind, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_envelope_enforces_size_and_terminal_shape() {
        let event = CodeModeEvent {
            identity: CodeModeIdentity {
                execution_id: "execution".into(),
                call_id: "outer".into(),
                control_request_id: 17,
                parent_call_id: None,
                session_id: "session".into(),
                agent_id: "agent".into(),
                trace_id: None,
            },
            phase: CodeModePhase::Started,
            request_id: None,
            summary: "started".into(),
            console: None,
            ok: None,
        };
        let UiEvent::Custom { kind, payload } = event.clone().into_ui_event().unwrap() else {
            panic!("expected Custom");
        };
        assert_eq!(kind, "code_mode.started");
        assert_eq!(
            serde_json::from_value::<CodeModeEvent>(payload).unwrap(),
            event
        );
        let mut invalid = event.clone();
        invalid.summary = "x".repeat(MAX_CODE_MODE_TEXT_BYTES + 1);
        assert!(invalid.into_ui_event().is_err());
        let mut invalid = event.clone();
        invalid.phase = CodeModePhase::Finished;
        assert!(invalid.into_ui_event().is_err());
        let mut finished = event;
        finished.phase = CodeModePhase::Finished;
        finished.ok = Some(false);
        assert!(finished.into_ui_event().is_ok());
    }
}
