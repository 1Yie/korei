//! `ask_question` — structured user input mid-turn: a question plus up to four
//! labelled options, rendered as a card in the approval slot. The call blocks
//! until the user answers or the turn is cancelled.
//!
//! `readonly: true` — asking mutates nothing, so it auto-allows in every mode.
//! Headless contexts carry no UI sender and refuse fast instead of deadlocking.

use super::registry::{Args, ToolCtx, ToolError, ToolResult, ToolSpec};
use futures::FutureExt;

const MAX_QUESTION_BYTES: usize = 1024;
const MAX_OPTIONS: usize = 4;
const MAX_LABEL_BYTES: usize = 80;
/// UI text constraint — chars, not bytes (a 中文 glyph is 3 UTF-8
/// bytes; the limit is what the card renders, not the payload size).
const MAX_DESCRIPTION_CHARS: usize = 200;

/// `schema_for` reads the doc comments into the JSON Schema descriptions.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct AskOptionArg {
    /// Short answer label rendered as a button (≤80 chars).
    label: String,
    /// Optional one-line explanation shown under the label.
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct AskQuestionArgs {
    /// The question to ask — one decision or gap, stated plainly.
    question: String,
    /// Up to 4 offered answers rendered as buttons. Omit for a pure
    /// free-text question.
    #[serde(default)]
    options: Vec<AskOptionArg>,
}

pub fn spec() -> ToolSpec {
    ToolSpec {
        name: "ask_question",
        schema: super::registry::schema_for::<AskQuestionArgs>(
            "Ask the user a structured question when a decision or missing piece of information \
             is blocking progress and cannot be reasonably inferred. The UI shows the question \
             with your options as buttons plus a free-text field; the tool blocks until the user \
             answers or cancels the turn. Use sparingly — prefer reasonable defaults, and never \
             ask questions you could answer by inspecting the environment yourself.",
        ),
        readonly: true, // asking mutates nothing; auto-allowed like the goal contract
        class: super::registry::ToolClass::HumanInteraction,
        network: false,
        staged: false,
        review: None,
        exec: std::sync::Arc::new(|args, ctx| exec(args, ctx).boxed()),
    }
}

async fn exec(args: Args, ctx: std::sync::Arc<ToolCtx>) -> Result<ToolResult, ToolError> {
    let parsed: AskQuestionArgs =
        serde_json::from_value(args).map_err(|e| ToolError::Args(e.to_string()))?;

    // Headless fast-fail before building the request — a child agent
    // must never park on a oneshot nobody can answer.
    let ask = ctx.ask()?;
    if !ask.is_available() {
        return Err(ToolError::Failed(
            "ask_question requires an interactive user session".into(),
        ));
    }
    let question = parsed.question.trim();
    if question.is_empty() || question.len() > MAX_QUESTION_BYTES {
        return Err(ToolError::Args(format!(
            "`question` must be 1..{MAX_QUESTION_BYTES} UTF-8 bytes"
        )));
    }
    if parsed.options.len() > MAX_OPTIONS {
        return Err(ToolError::Args(format!(
            "at most {MAX_OPTIONS} options — merge or drop the rest"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    let mut options = Vec::with_capacity(parsed.options.len());
    for opt in &parsed.options {
        let label = opt.label.trim();
        if label.is_empty() || label.len() > MAX_LABEL_BYTES {
            return Err(ToolError::Args(format!(
                "option labels must be 1..{MAX_LABEL_BYTES} bytes"
            )));
        }
        if !seen.insert(label.to_string()) {
            return Err(ToolError::Args("option labels must be unique".into()));
        }
        let description = match opt.description.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(t) if t.chars().count() > MAX_DESCRIPTION_CHARS => {
                return Err(ToolError::Args(format!(
                    "option descriptions must be ≤{MAX_DESCRIPTION_CHARS} chars"
                )));
            }
            Some(t) => Some(t.to_string()),
        };
        options.push(agent_ipc::events::AskOption {
            label: label.to_string(),
            description,
        });
    }

    let answer = ask
        .ask(question.to_string(), options.clone(), ctx.cancel.clone())
        .await?;
    // Echo which structured option the answer maps to — the UI knows
    // it was a click vs free text, and telemetry can consume the index
    // without re-parsing prose.
    let result = match options.iter().position(|o| o.label == answer) {
        Some(i) => format!("用户回答：{answer}（选项 {i}）"),
        None => format!("用户回答（自定义）：{answer}"),
    };
    Ok(ToolResult::text(result))
}
