//! `LlmProvider` trait + `BoxStream` alias.
//!
//! `Send + Sync` + `Arc` — the kernel swaps providers by replacing the `Arc`,
//! which is what makes hot-switching free (applies next turn, never
//! interrupts an in-flight stream).

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::compat::ProviderCompat;
use crate::types::{ChatMessage, StreamChunk};

pub type BoxStream<T> = Pin<Box<dyn Stream<Item = anyhow::Result<T>> + Send>>;

/// Per-model wire settings resolved from config and carried on every request.
///
/// The provider instance owns *provider-level* configuration (base URL, key,
/// provider-level `compat`, headers) because it is built once from
/// `ProviderConfig`; these are the parts that vary per **model**, so they ride
/// the request instead. `None`/empty means "the model declares nothing" — the
/// adapter's own default applies, which is what keeps an unconfigured model
/// behaving exactly as it did before this existed.
#[derive(Debug, Clone, Default)]
pub struct ModelParams {
    /// Output-token cap (pi `maxTokens`). `None` → the adapter's default cap.
    pub max_tokens: Option<u32>,
    /// Sampling parameters merged verbatim into the request body *after* the
    /// fields the adapter sets itself, so these keys win (pi `samplingParams`).
    /// Only OpenAI-shaped APIs apply it.
    pub sampling_params: Option<serde_json::Value>,
    /// The model's `compat`, already merged over the provider's for the fields
    /// it declares. Adapters consult this *instead of* their stored
    /// provider-level compat wherever the value is model-specific.
    pub compat: Option<ProviderCompat>,
}

impl ModelParams {
    /// No per-model settings — the adapter's own defaults apply everywhere.
    /// Used by tests and by any caller that has no config entry for the model.
    pub const EMPTY: ModelParams = ModelParams {
        max_tokens: None,
        sampling_params: None,
        compat: None,
    };

    /// The effective compat for this request — the provider's own values with
    /// the model's merged over them.
    pub fn compat_with(&self, provider: Option<&ProviderCompat>) -> ProviderCompat {
        provider
            .cloned()
            .unwrap_or_default()
            .merged_with(self.compat.as_ref())
    }

    /// Merge `samplingParams` into an assembled request body. Keys win over
    /// anything the adapter set, matching pi ("its keys win").
    pub fn apply_sampling_params(&self, body: &mut serde_json::Value) {
        let Some(extra) = self.sampling_params.as_ref().and_then(|v| v.as_object()) else {
            return;
        };
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in extra {
                obj.insert(k.clone(), v.clone());
            }
        }
    }

    /// Write the output-token cap into `body` under `field`, clamped to
    /// `max_tokens.min(u32::MAX)`. No-op when the model declares no cap.
    pub fn apply_max_tokens(&self, body: &mut serde_json::Value, field: &str) {
        if let Some(max) = self.max_tokens {
            body[field] = serde_json::json!(max);
        }
    }
}

/// What a wire protocol can actually express — declared by the adapter as
/// a baseline, narrowed per-deployment by `ProviderCompat` config fields.
/// The engine consults this BEFORE assembling request params, so rules
/// like "no temperature while thinking" live in one declared table instead
/// of scattering `if provider == …` guesses across adapters.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    /// Structured tool calls on the wire (OpenAI `tool_calls`, Anthropic
    /// `tool_use`, Gemini `functionCall`). `false` = a text-protocol
    /// provider — the kernel flattens calls into message text instead.
    pub native_tool_calls: bool,
    /// Accepts a reasoning/thinking directive (`reasoning_effort`,
    /// `thinking.budget_tokens`, `thinkingConfig.thinkingBudget`). `false`
    /// → the engine never computes an effort value for this provider.
    pub reasoning: bool,
    /// `temperature` may be sent while reasoning is enabled. Anthropic's
    /// thinking mode rejects temperature outright (400); OpenAI accepts
    /// both — adapters that return `false` here drop the field themselves.
    pub temperature_with_reasoning: bool,
    /// Accepts inline image parts (`input_image` / `image` / `inlineData`).
    pub vision: bool,
}

impl Capabilities {
    /// Everything-supported baseline — adapters narrow what their wire
    /// cannot express.
    pub const fn all() -> Self {
        Self {
            native_tool_calls: true,
            reasoning: true,
            temperature_with_reasoning: true,
            vision: true,
        }
    }
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Stable adapter id: `"openai_completions" | "anthropic" | "openai_responses" | "gemini"`.
    fn id(&self) -> &'static str;

    /// Baseline capability table — adapters override; `ProviderCompat`
    /// config fields can narrow it further per-deployment. See field docs
    /// for what each bit gates.
    fn capabilities(&self) -> Capabilities {
        Capabilities::all()
    }

    /// Whether this provider consumes structured `tool_calls` on the wire (OpenAI
    /// `function_call`, Anthropic `tool_use`).
    ///
    /// `false` marks a text-protocol provider that only understands inline
    /// `[call: …]` echoes; the sanitize step then flattens `tool_calls` into message
    /// text. Flattening a native provider's `tool_calls` would orphan its
    /// `Role::Tool` results (P1-b).
    fn native_tool_calls(&self) -> bool {
        self.capabilities().native_tool_calls
    }

    /// `tools` is a provider-agnostic JSON Schema array; the adapter maps it
    /// onto its wire shape. `temperature` is a soft hint — adapters may clamp.
    /// `reasoning_effort` is an optional hint ("low", "medium", "high", "max") for models supporting reasoning.
    /// `params` carries the per-model wire settings (`maxTokens`,
    /// `samplingParams`, model-level `compat`) resolved from config.
    async fn chat_stream(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: Option<serde_json::Value>,
        temperature: f32,
        reasoning_effort: Option<&str>,
        params: &ModelParams,
    ) -> anyhow::Result<BoxStream<StreamChunk>>;
}

/// Stand-in when `config.toml` names no usable provider: every stream call
/// fails immediately with a clear message instead of silently answering
/// with canned output. `models()` is empty so pickers show nothing.
pub struct UnconfiguredProvider;

#[async_trait]
impl LlmProvider for UnconfiguredProvider {
    fn id(&self) -> &'static str {
        "unconfigured"
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
        _tools: Option<serde_json::Value>,
        _temperature: f32,
        _reasoning_effort: Option<&str>,
        _params: &ModelParams,
    ) -> anyhow::Result<BoxStream<StreamChunk>> {
        anyhow::bail!("no provider configured — add one in Settings")
    }
}

/// Stand-in for a configured provider that could not be built (missing key,
/// unset env var, keyring miss). Like `UnconfiguredProvider` it never answers,
/// but it carries the reason so the failed turn tells the user what to fix.
pub struct UnavailableProvider {
    reason: String,
}

impl UnavailableProvider {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl LlmProvider for UnavailableProvider {
    fn id(&self) -> &'static str {
        "unavailable"
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: &[ChatMessage],
        _tools: Option<serde_json::Value>,
        _temperature: f32,
        _reasoning_effort: Option<&str>,
        _params: &ModelParams,
    ) -> anyhow::Result<BoxStream<StreamChunk>> {
        anyhow::bail!("provider unavailable: {}", self.reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A provider that failed to build must say WHY, and must fail the turn.
    /// It must never answer, and the reason must reach the user verbatim.
    #[tokio::test]
    async fn unavailable_provider_fails_with_the_real_reason() {
        let p = UnavailableProvider::new("env var OPENAI_API_KEY unset or empty");
        let err = match p
            .chat_stream("m", &[], None, 0.0, None, &ModelParams::EMPTY)
            .await
        {
            Ok(_) => panic!("unavailable provider must not stream"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("OPENAI_API_KEY"), "{err}");
    }
}
