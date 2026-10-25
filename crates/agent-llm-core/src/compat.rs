//! `ProviderCompat` — per-provider / per-model wire-dialect flags (pure data).

use serde::{Deserialize, Serialize};

/// Feature compatibility flags — the full pi (`models.json`) `compat` surface.
///
/// Every field is optional; `None` means "not declared" and the adapter's default
/// applies. Set per provider or per model, the model winning. Rust snake_case names
/// accept the pi camelCase spelling as an alias, so `models.json` and a
/// settings-written `config.toml` both round-trip.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderCompat {
    /// Whether this model accepts temperature at all. None keeps the adapter's
    /// existing default; false vetoes both caller hints and sampling overrides.
    #[serde(
        default,
        alias = "supportsTemperature",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_temperature: Option<bool>,

    // ---- OpenAI-compatible (openai-completions / openai-responses) ----
    /// Provider accepts the `store` request field.
    #[serde(
        default,
        alias = "supports_store",
        alias = "supportsStore",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_store: Option<bool>,
    /// Send the system prompt as `developer` (true) or `system` (false).
    #[serde(
        default,
        alias = "supports_developer_role",
        alias = "supportsDeveloperRole",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_developer_role: Option<bool>,
    /// Provider accepts `reasoning_effort`. `false` is the per-deployment kill
    /// switch — the engine then never computes an effort value at all.
    #[serde(
        default,
        alias = "supports_reasoning_effort",
        alias = "supportsReasoningEffort",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_reasoning_effort: Option<bool>,
    /// Output-token cap field: `max_completion_tokens` or `max_tokens`.
    /// `None` → the adapter's own default.
    #[serde(
        default,
        alias = "max_tokens_field",
        alias = "maxTokensField",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_tokens_field: Option<String>,
    /// Provider accepts `stream_options: { include_usage: true }` (pi default true).
    #[serde(
        default,
        alias = "supports_usage_in_streaming",
        alias = "supportsUsageInStreaming",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_usage_in_streaming: Option<bool>,
    /// Streamed responses carry `finish_reason`; when false the adapter infers
    /// `stop` / `toolUse` from stream termination (pi default true).
    #[serde(
        default,
        alias = "supports_finish_reason",
        alias = "supportsFinishReason",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_finish_reason: Option<bool>,
    /// Include `name` on tool-result messages.
    #[serde(
        default,
        alias = "requires_tool_result_name",
        alias = "requiresToolResultName",
        skip_serializing_if = "Option::is_none"
    )]
    pub requires_tool_result_name: Option<bool>,
    /// Insert an assistant message between tool results and a following user message.
    #[serde(
        default,
        alias = "requires_assistant_after_tool_result",
        alias = "requiresAssistantAfterToolResult",
        skip_serializing_if = "Option::is_none"
    )]
    pub requires_assistant_after_tool_result: Option<bool>,
    /// Convert thinking blocks to plain text instead of provider-native reasoning.
    #[serde(
        default,
        alias = "requires_thinking_as_text",
        alias = "requiresThinkingAsText",
        skip_serializing_if = "Option::is_none"
    )]
    pub requires_thinking_as_text: Option<bool>,
    /// Replay an empty `reasoning_content` on every assistant message while
    /// reasoning is enabled (some backends require the key to be present).
    #[serde(
        default,
        alias = "requires_reasoning_content_on_assistant_messages",
        alias = "requiresReasoningContentOnAssistantMessages",
        skip_serializing_if = "Option::is_none"
    )]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    /// Thinking-parameter dialect: `openai`, `openrouter`, `deepseek`,
    /// `together`, `baseten`, `zai`, `qwen`, `chat-template`,
    /// `qwen-chat-template`, `string-thinking`, `ant-ling`.
    #[serde(
        default,
        alias = "thinking_format",
        alias = "thinkingFormat",
        skip_serializing_if = "Option::is_none"
    )]
    pub thinking_format: Option<String>,
    /// `chat_template_kwargs` values for `thinkingFormat: "chat-template"`.
    #[serde(
        default,
        alias = "chat_template_kwargs",
        alias = "chatTemplateKwargs",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// `chat_template_args` values for `thinkingFormat: "baseten"`.
    #[serde(
        default,
        alias = "chat_template_args",
        alias = "chatTemplateArgs",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_template_args: Option<serde_json::Value>,
    /// Top-level request field capping reasoning tokens: `thinking_token_budget`
    /// (vLLM), `thinking_budget` (Qwen/DashScope/SGLang),
    /// `thinking_budget_tokens` (llama.cpp).
    #[serde(
        default,
        alias = "thinking_token_budget_field",
        alias = "thinkingTokenBudgetField",
        skip_serializing_if = "Option::is_none"
    )]
    pub thinking_token_budget_field: Option<String>,
    /// Alias for `thinking_token_budget_field` — `true` means the default field.
    #[serde(
        default,
        alias = "supports_thinking_token_budget",
        alias = "supportsThinkingTokenBudget",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_thinking_token_budget: Option<bool>,
    /// Anthropic-style `cache_control` markers on an OpenAI-compatible wire.
    /// Only `"anthropic"` is meaningful.
    #[serde(
        default,
        alias = "cache_control_format",
        alias = "cacheControlFormat",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_control_format: Option<String>,
    /// Send session-affinity headers derived from the session id.
    #[serde(
        default,
        alias = "send_session_affinity_headers",
        alias = "sendSessionAffinityHeaders",
        skip_serializing_if = "Option::is_none"
    )]
    pub send_session_affinity_headers: Option<bool>,
    /// Session-affinity header dialect: `openai`, `openai-nosession`, `openrouter`.
    #[serde(
        default,
        alias = "session_affinity_format",
        alias = "sessionAffinityFormat",
        skip_serializing_if = "Option::is_none"
    )]
    pub session_affinity_format: Option<String>,
    /// Provider accepts strict JSON-schema function tool definitions.
    #[serde(
        default,
        alias = "supports_strict_mode",
        alias = "supportsStrictMode",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_strict_mode: Option<bool>,
    /// Emit custom Lark/regex grammar tools (GPT-5+ on OpenAI and friends).
    #[serde(
        default,
        alias = "supports_openai_grammar_tools",
        alias = "supportsOpenAIGrammarTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_openai_grammar_tools: Option<bool>,
    /// Accept long cache retention when cache retention is `long`.
    #[serde(
        default,
        alias = "supports_long_cache_retention",
        alias = "supportsLongCacheRetention",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_long_cache_retention: Option<bool>,
    /// OpenRouter provider-routing preferences — sent verbatim as the request
    /// `provider` field.
    #[serde(
        default,
        alias = "open_router_routing",
        alias = "openRouterRouting",
        skip_serializing_if = "Option::is_none"
    )]
    pub open_router_routing: Option<serde_json::Value>,
    /// Vercel AI Gateway routing config (`only`, `order`).
    #[serde(
        default,
        alias = "vercel_gateway_routing",
        alias = "vercelGatewayRouting",
        skip_serializing_if = "Option::is_none"
    )]
    pub vercel_gateway_routing: Option<serde_json::Value>,

    // ---- Anthropic Messages (anthropic-messages) ----
    /// Per-tool `eager_input_streaming: true` (pi default true). `false` omits
    /// the field and sends the legacy fine-grained-tool-streaming beta header.
    #[serde(
        default,
        alias = "supports_eager_tool_input_streaming",
        alias = "supportsEagerToolInputStreaming",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_eager_tool_input_streaming: Option<bool>,
    /// `cache_control.ttl: "1h"` on tool definitions is accepted.
    #[serde(
        default,
        alias = "supports_cache_control_on_tools",
        alias = "supportsCacheControlOnTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_cache_control_on_tools: Option<bool>,
    /// Send adaptive thinking (`thinking.type: "adaptive"` + `output_config.effort`).
    #[serde(
        default,
        alias = "force_adaptive_thinking",
        alias = "forceAdaptiveThinking",
        skip_serializing_if = "Option::is_none"
    )]
    pub force_adaptive_thinking: Option<bool>,
    /// Adapter-specific per-turn effort support. Responses uses
    /// configuration_update items; Anthropic uses its own messages/beta
    /// controls. Declaring one does not make the other protocol valid.
    #[serde(
        default,
        alias = "supports_mid_convo_effort",
        alias = "supportsMidConvoEffort",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_mid_convo_effort: Option<bool>,
    /// Replay empty thinking signatures as `signature: ""` instead of turning
    /// thinking into text — only for providers that emit empty signatures.
    #[serde(
        default,
        alias = "allow_empty_signature",
        alias = "allowEmptySignature",
        skip_serializing_if = "Option::is_none"
    )]
    pub allow_empty_signature: Option<bool>,
    /// Provider accepts strict JSON-schema tool definitions (Anthropic flavour).
    #[serde(
        default,
        alias = "supports_strict_tools",
        alias = "supportsStrictTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_strict_tools: Option<bool>,
    /// Up to three server-side fallback models, each `{provider, model, cost}`.
    /// An empty array disables fallback.
    #[serde(
        default,
        alias = "allowed_fallback_models",
        alias = "allowedFallbackModels",
        skip_serializing_if = "Option::is_none"
    )]
    pub allowed_fallback_models: Option<serde_json::Value>,
}

/// Every optional field of `ProviderCompat`, for field-wise merge — a macro
/// keeps the list in one place so a new field is never silently un-merged.
macro_rules! compat_fields {
    ($mac:ident) => {
        $mac! {
            supports_temperature,
            supports_store,
            supports_developer_role,
            supports_reasoning_effort,
            max_tokens_field,
            supports_usage_in_streaming,
            supports_finish_reason,
            requires_tool_result_name,
            requires_assistant_after_tool_result,
            requires_thinking_as_text,
            requires_reasoning_content_on_assistant_messages,
            thinking_format,
            chat_template_kwargs,
            chat_template_args,
            thinking_token_budget_field,
            supports_thinking_token_budget,
            cache_control_format,
            send_session_affinity_headers,
            session_affinity_format,
            supports_strict_mode,
            supports_openai_grammar_tools,
            supports_long_cache_retention,
            open_router_routing,
            vercel_gateway_routing,
            supports_eager_tool_input_streaming,
            supports_cache_control_on_tools,
            force_adaptive_thinking,
            supports_mid_convo_effort,
            allow_empty_signature,
            supports_strict_tools,
            allowed_fallback_models,
        }
    };
}

impl ProviderCompat {
    /// Field-wise merge: every field `over` declares replaces `self`'s; fields
    /// it leaves `None` keep the base value. This is pi's semantic for
    /// provider `compat` → model `compat`.
    pub fn merged_with(&self, over: Option<&ProviderCompat>) -> ProviderCompat {
        let Some(over) = over else {
            return self.clone();
        };
        let mut out = self.clone();
        macro_rules! take {
            ($($f:ident),* $(,)?) => {
                $(if over.$f.is_some() { out.$f = over.$f.clone(); })*
            };
        }
        compat_fields!(take);
        out
    }

    /// Whether the system prompt should be sent with the `developer` role.
    /// OpenAI's own default is `developer` for reasoning models; a proxy that
    /// only knows `system` sets `supportsDeveloperRole: false` (pi semantics).
    pub fn developer_role(&self) -> bool {
        self.supports_developer_role.unwrap_or(false)
    }

    /// The output-token cap field name — `max_completion_tokens` unless the
    /// provider declares `max_tokens` (pi's `maxTokensField`).
    pub fn max_tokens_field_name(&self) -> &str {
        self.max_tokens_field
            .as_deref()
            .unwrap_or("max_completion_tokens")
    }

    /// The reasoning-token budget field, resolving the
    /// `supportsThinkingTokenBudget` alias (`true` → the default field name).
    pub fn thinking_token_budget_field_name(&self) -> Option<&str> {
        if let Some(f) = self.thinking_token_budget_field.as_deref() {
            return Some(f);
        }
        self.supports_thinking_token_budget
            .filter(|enabled| *enabled)
            .map(|_| "thinking_token_budget")
    }

    /// The effective thinking dialect (`openai` when undeclared).
    pub fn thinking_format_name(&self) -> &str {
        self.thinking_format.as_deref().unwrap_or("openai")
    }
}
