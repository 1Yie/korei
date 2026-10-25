//! Normalized model — the only types that cross the provider boundary.
//!
//! * `arguments` fragments are concatenated, never parsed until `Done` or the next
//!   slot begins — JSON arrives split across chunks.
//! * `Done` fires exactly once per request, whatever the backend's sentinel.
//! * `Error` is a stream item, not stream termination — `SamplerActor` decides
//!   retry vs propagate.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A user-attached image on a `Role::User` message. Persisted as a path
/// reference — the file is staged inside the workspace (`.husk/attachments/`),
/// so it is visible to sandboxed tools AND durable for session replay.
/// Adapters materialize the bytes into a `data:` URL at wire-build time;
/// never inline the payload into the message itself (the session snapshot
/// would balloon to megabytes per image).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRef {
    /// Absolute path — the staged copy inside the workspace.
    pub path: PathBuf,
    /// MIME type (`image/png`, `image/jpeg`, …) — sniffed from the
    /// extension, good enough for `data:` URL framing.
    pub media_type: String,
}

impl ImageRef {
    /// Build a ref for `path` — `None` when the extension isn't an image
    /// type we can frame.
    pub fn for_path(path: PathBuf) -> Option<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())?;
        let media_type = match ext.as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "bmp" => "image/bmp",
            _ => return None,
        };
        Some(Self {
            path,
            media_type: media_type.into(),
        })
    }

    /// Read the file and frame it as `data:<mime>;base64,<bytes>` — the
    /// shape every vision-capable wire protocol accepts. `None` when the
    /// file is gone or over the provider-size bound (20MB is the widest
    /// common cap — Anthropic 5MB is tighter but rare).
    pub fn data_url(&self) -> Option<String> {
        const MAX_IMAGE_BYTES: u64 = 20 * 1024 * 1024;
        let meta = std::fs::metadata(&self.path).ok()?;
        if meta.len() > MAX_IMAGE_BYTES {
            return None;
        }
        let bytes = std::fs::read(&self.path).ok()?;
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Some(format!("data:{};base64,{}", self.media_type, b64))
    }
}

/// How a persisted message replays in the UI. On `Role::System` entries,
/// `Some` marks a line the live stream actually showed (`SystemMessage` /
/// `Error` events) while `None` is internal context (system prompt,
/// compaction note, hook injection) — hidden. `Hidden` marks a message
/// the provider must see but the UI never rendered (injected
/// instructions) — without it a reloaded view invents a user bubble the
/// user never typed. Adapters strip this before the wire like
/// [`ChatMessage::is_error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoticeKind {
    /// Plain system line — e.g. "已被用户中断", a `/` command reply.
    System,
    /// `⚠`-prefixed error line — what the `UiEvent::Error` arm renders.
    Error,
    /// Persisted for the provider but never drawn — injected
    /// instructions like the tool-limit nudge.
    Hidden,
    /// Compacted-context memory note — a summary of earlier turns. Still
    /// `Role::System` internally (hidden on replay like the kernel
    /// prompt), but adapters MUST NOT fold it into `system`/`instructions`
    /// like a plain system row: it is historical context, not a live
    /// instruction. Each adapter emits it at user privilege instead.
    CompactedMemory,
    /// UI-only compaction card row (`ChatMessage::compaction`) — the
    /// persisted before/after accounting the frontend replays as a
    /// compaction card. Dropped from every provider wire body: it is
    /// render metadata, never context for the model.
    Compacted,
    /// UI-only product card (`ChatMessage::card`). Persisted as its kind
    /// string (`"plan"`), so earlier snapshots read back unchanged. Adapters
    /// drop it from the wire like [`NoticeKind::Compacted`].
    Card(String),
    /// A mid-turn user steer (`ChatMessage::user_steer`). The model sees an
    /// ordinary user message; the tag lets replay and counters tell it from
    /// the prompt that opened the turn, which it belongs to. Unlike `Hidden`
    /// it is drawn — as a user bubble inside the running turn.
    Steer,
    /// A host-pushed notice (`ChatMessage::host_notice`) — a background job
    /// finishing, a watch firing. Like [`NoticeKind::CompactedMemory`] it is
    /// `Role::System` internally (never a user turn, never a steer) but
    /// adapters MUST NOT fold it into `system`/`instructions`: the text can
    /// carry untrusted tool output, so it is emitted at user privilege,
    /// fenced as a host notice. The UI shows the event behind it through
    /// its own channel, so replay does not draw the row.
    Host,
}

impl NoticeKind {
    const NAMES: [(&'static str, NoticeKind); 7] = [
        ("system", NoticeKind::System),
        ("error", NoticeKind::Error),
        ("hidden", NoticeKind::Hidden),
        ("compacted_memory", NoticeKind::CompactedMemory),
        ("compacted", NoticeKind::Compacted),
        ("steer", NoticeKind::Steer),
        ("host", NoticeKind::Host),
    ];

    /// Render metadata only — never context for the model.
    pub fn is_ui_only(&self) -> bool {
        matches!(self, NoticeKind::Compacted | NoticeKind::Card(_))
    }
}

impl Serialize for NoticeKind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            NoticeKind::Card(kind) => s.serialize_str(kind),
            known => {
                let name = Self::NAMES
                    .iter()
                    .find(|(_, k)| k == known)
                    .map(|(n, _)| *n);
                s.serialize_str(name.expect("every non-card notice is named"))
            }
        }
    }
}

/// Reserved names map to their variants; any other string is a card kind.
impl<'de> Deserialize<'de> for NoticeKind {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        Ok(Self::NAMES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, k)| k.clone())
            .unwrap_or(NoticeKind::Card(name)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Opaque, replay-only state owned by an adapter, not user-visible reasoning.
/// The kernel persists and forwards it without interpreting provider JSON.
/// Adapters must verify both origin and model before replaying its data.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderState {
    pub origin: String,
    pub model: String,
    pub data: String,
}

impl std::fmt::Debug for ProviderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderState")
            .field("origin", &self.origin)
            .field("model", &self.model)
            .field("data_bytes", &self.data.len())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// `None` serializes as `""` on the wire — some OpenAI-compat backends
    /// (devin verified) treat a literal `content:null` on an assistant
    /// tool-call message as malformed and return an empty stream instead of
    /// an error, which surfaces as "the agent went quiet after a tool".
    /// The reverse mapping is `""` → `None`, not `Some("")`: a snapshot
    /// round-trip then preserves the empty-content invariant the resume
    /// sanitizer and every `is_empty` guard already rely on.
    #[serde(
        serialize_with = "content_as_string",
        deserialize_with = "empty_string_as_none"
    )]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Replay-only metadata: this tool result carried a failure (veto,
    /// deny, dispatch error, write failure, …). Persisted in the session
    /// snapshot so a reloaded view can re-render the failed capsule —
    /// without it every tool call replays as `ok`. Adapters that forward
    /// `ChatMessage` raw (`openai_completions`) must strip it before the wire;
    /// strict backends reject unknown fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Display class for replay — see [`NoticeKind`]. `None` renders by
    /// role (system = internal/hidden, others = their normal item).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notice: Option<NoticeKind>,
    /// Creation time (epoch ms) — persisted so a replayed view can draw
    /// the same `—— time ——` turn divider the live stream showed. `None`
    /// on old snapshots → no divider for those messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ts: Option<i64>,
    /// User-attached images — staged path references, materialized into
    /// `data:` URLs by each adapter at wire-build time. Only ever set on
    /// `Role::User` messages and only when the active model declares
    /// `"image"` in its `input` modalities; empty otherwise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageRef>,
    /// Reasoning/thinking trace of this assistant round — the persisted form
    /// of the live `思考过程` block, so a reopened session replays what the
    /// stream actually drew. Display-only: `reasoning` is a *separate*
    /// reasoning-model input, never a transcript field, so every adapter must
    /// keep it off the wire (`openai_completions` strips it like `is_error`).
    /// `None` on old snapshots and on rounds that produced no trace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Wall-clock duration (ms) of the activity this message records —
    /// `role=tool`: the call's dispatch-to-result span; `role=assistant`:
    /// the reasoning block's first→last delta span. Persisted so a replayed
    /// view can re-render the `思考过程 · 20s` / `工具调用 · 3s` elapsed
    /// labels the live stream drew. Display-only like `reasoning` — every
    /// adapter must keep it off the wire (`openai_completions` strips it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    /// Adapter-owned continuity data (e.g. encrypted reasoning). Never render
    /// it as text or send it to a different provider/model. Absent on old logs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_state: Option<ProviderState>,
}

/// Wall-clock epoch millis — stamps every message at construction.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn content_as_string<S: serde::Serializer>(c: &Option<String>, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(c.as_deref().unwrap_or(""))
}

/// Paired with `content_as_string` — the persisted `""` means `None`,
/// never `Some("")`. (A genuinely-empty `Some("")` collapses to `None`
/// too, which is semantically identical everywhere the value is read.)
fn empty_string_as_none<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<String>, D::Error> {
    let s = String::deserialize(d)?;
    Ok(if s.is_empty() { None } else { Some(s) })
}

impl ChatMessage {
    pub fn with_provider_state(mut self, state: Option<ProviderState>) -> Self {
        self.provider_state = state;
        self
    }

    /// Attach this round's reasoning trace. Empty/whitespace traces are
    /// dropped rather than persisted as an empty thought block.
    pub fn with_reasoning(mut self, reasoning: Option<String>) -> Self {
        self.reasoning = reasoning.filter(|r| !r.trim().is_empty());
        self
    }

    /// Stamp the activity duration — reasoning span on assistant rows,
    /// dispatch span on tool results.
    pub fn with_duration_ms(mut self, ms: Option<i64>) -> Self {
        self.duration_ms = ms;
        self
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: None,
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: None,
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// Attach staged image refs — only call this when the active model
    /// declares `"image"` in its `input` modalities.
    pub fn with_images(mut self, images: Vec<ImageRef>) -> Self {
        self.images = images;
        self
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: None,
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// A user-facing system line the live stream emitted via
    /// `UiEvent::SystemMessage` — persisted so a reloaded view replays
    /// it exactly (vs `system()`, which is invisible internal context).
    pub fn notice(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: Some(NoticeKind::System),
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// A host-pushed notice — see [`NoticeKind::Host`]. Model-facing: the
    /// text may end in a log tail meant for the model only.
    pub fn host_notice(text: impl Into<String>) -> Self {
        Self {
            notice: Some(NoticeKind::Host),
            ..Self::notice(text)
        }
    }
    /// Same, for a `UiEvent::Error` line — replays with the `⚠` prefix.
    pub fn notice_error(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: Some(NoticeKind::Error),
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// A `Role::User` instruction the UI never showed (injected by the
    /// engine, e.g. the synthesis nudge) — kept for the provider,
    /// skipped on replay.
    pub fn user_hidden(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: Some(NoticeKind::Hidden),
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// Text the model sees in front of a mid-turn steer. Part of the prompt,
    /// so it is also how a row persisted before [`NoticeKind::Steer`] existed
    /// is recognised ([`ChatMessage::is_steer`]).
    pub const STEER_PREFIX: &'static str = "The user interrupted: ";

    /// A `Role::User` correction injected while a turn is running. The model
    /// reads `The user interrupted: <text>`; replay and turn counters read
    /// the [`NoticeKind::Steer`] tag.
    pub fn user_steer(text: impl AsRef<str>) -> Self {
        Self {
            role: Role::User,
            content: Some(format!("{}{}", Self::STEER_PREFIX, text.as_ref())),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: Some(NoticeKind::Steer),
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }

    /// Whether this row is a mid-turn steer. Rows written before the tag
    /// existed carry no `notice`, only the prefix, so an untagged user row is
    /// matched on it; a tagged row (`Hidden`, a card, …) never is — a hidden
    /// instruction that happens to start with the prefix stays hidden.
    pub fn is_steer(&self) -> bool {
        self.role == Role::User
            && match &self.notice {
                Some(NoticeKind::Steer) => true,
                None => self
                    .content
                    .as_deref()
                    .is_some_and(|c| c.starts_with(Self::STEER_PREFIX)),
                Some(_) => false,
            }
    }

    /// The steer's own words, without the prefix the model reads. `None` for
    /// a row that is not a steer.
    pub fn steer_text(&self) -> Option<&str> {
        if !self.is_steer() {
            return None;
        }
        let content = self.content.as_deref()?;
        Some(content.strip_prefix(Self::STEER_PREFIX).unwrap_or(content))
    }

    /// The compaction memory note — kept as `Role::System` so replay
    /// stays hidden, but tagged so adapters route it to user privilege
    /// instead of folding it into `system`/`instructions`.
    pub fn compacted_memory(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: Some(NoticeKind::CompactedMemory),
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// The persisted compaction card — a JSON payload (`before_tokens`,
    /// `after_tokens`, `removed_messages`, `manual`, `note`) the frontend
    /// renders as the compaction card on replay. `NoticeKind::Compacted`
    /// rows are dropped by every adapter, so the note here never costs wire
    /// tokens; the model sees the summary through the `compacted_memory`
    /// row. `manual` keeps the user-run `/compact` card a standalone block
    /// on replay instead of folding it into the previous turn.
    pub fn compaction(
        before_tokens: u32,
        after_tokens: u32,
        removed_messages: u32,
        manual: bool,
        note: &str,
    ) -> Self {
        let payload = serde_json::json!({
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "removed_messages": removed_messages,
            "manual": manual,
            "note": note,
        });
        Self {
            role: Role::System,
            content: Some(payload.to_string()),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: Some(NoticeKind::Compacted),
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// A persisted UI-only card of `kind` whose `payload` the host's frontend
    /// renders on replay. Adapters drop it: it is never model context.
    pub fn card(kind: impl Into<String>, payload: String) -> Self {
        Self {
            role: Role::System,
            content: Some(payload),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: Some(NoticeKind::Card(kind.into())),
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    pub fn tool_result(call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            is_error: None,
            notice: None,
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
    /// Attach staged image refs a *tool* produced (a screenshot) — the tool
    /// analogue of [`ChatMessage::with_images`]. Adapters encode them per
    /// their wire: an Anthropic `tool_result` carries real image blocks, the
    /// protocols whose tool output is text-only emit a synthetic user turn
    /// right after (see each adapter's `build_messages`).
    pub fn tool_result_with_images(mut self, images: Vec<ImageRef>) -> Self {
        self.images = images;
        self
    }
    /// Failed tool result — same wire shape, plus the persisted `is_error`
    /// flag the UI replays into the red capsule state.
    pub fn tool_result_err(call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            is_error: Some(true),
            notice: None,
            ts: Some(now_ms()),
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Canonical invocation id — provider-issued when the wire has one
    /// (`call_…`, `toolu_…`), `call_*`-synthesized by the assembler when it
    /// doesn't (Gemini `functionCall` carries no id at all). Never empty
    /// once a call reaches the engine. Adapters replay it through their
    /// own correlation field (`call_id`, `tool_use_id`); name-keyed wires
    /// (Gemini `functionResponse`) resolve the tool name by looking this
    /// id up against the earlier assistant `tool_calls`.
    pub id: String,
    pub name: String,
    /// JSON assembled incrementally from `args_delta` fragments.
    pub arguments: String,
}

// OpenAI wire shape: `{"id":…,"type":"function","function":{"name":…,
// "arguments":"{…}"}}` — NOT our flat `{id,name,arguments}`. A tool-call
// round-trip serialized flat makes the next request's assistant message
// malformed and the model stops responding after the first tool call
// (verified against devin/swe-2).
impl Serialize for ToolCall {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("ToolCall", 3)?;
        st.serialize_field("id", &self.id)?;
        st.serialize_field("type", "function")?;
        st.serialize_field(
            "function",
            &serde_json::json!({
                "name": self.name,
                "arguments": self.arguments,
            }),
        )?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Accept both the OpenAI shape and the flat shape on the way in.
        #[derive(Deserialize)]
        struct Flat {
            id: String,
            name: String,
            arguments: String,
        }
        #[derive(Deserialize)]
        struct Func {
            name: String,
            arguments: String,
        }
        #[derive(Deserialize)]
        struct OpenAi {
            id: String,
            function: Func,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Any {
            Oa(OpenAi),
            Fl(Flat),
        }
        match Any::deserialize(d)? {
            Any::Oa(o) => Ok(ToolCall {
                id: o.id,
                name: o.function.name,
                arguments: o.function.arguments,
            }),
            Any::Fl(f) => Ok(ToolCall {
                id: f.id,
                name: f.name,
                arguments: f.arguments,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StreamChunk {
    /// Reasoning trace (DeepSeek-R1 `reasoning_content`, Claude thinking,
    /// Grok reasoning).
    ReasoningDelta(String),
    /// Replay-only adapter state for a successfully completed assistant round.
    /// Persist on its message; never render it or count it as generated text.
    ProviderState(ProviderState),
    /// User-visible text delta.
    ContentDelta(String),
    /// Tool-call fragment — id/name arrive once, arguments stream as JSON shards under
    /// `args_delta`.
    ///
    /// `slot` is an opaque merge key for which invocation a delta belongs to, not a
    /// positional index: OpenAI sends `tool_calls[i].index`, Responses a sparse
    /// `output_index`, Anthropic a content-block `index`, Gemini a counter. The
    /// assembler only needs deltas of one invocation to share a slot, and slots to
    /// sort in call order.
    ToolCallDelta {
        slot: usize,
        id: Option<String>,
        name: Option<String>,
        args_delta: String,
    },
    /// Terminal chunk; carries usage when the backend reports it.
    Done {
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
        /// Prompt tokens served from the provider's prompt cache —
        /// `prompt - cached` is the fresh/billed portion.
        cached_tokens: Option<u32>,
    },
    Error(String),
    /// The upstream is alive but has produced nothing to show — a keep-alive
    /// ping, an SSE comment, an event the provider does not surface. It carries
    /// no content and is not output: the sampler uses it to restart its idle
    /// clock and never forwards it, so a model that thinks silently for minutes
    /// is not mistaken for a wedged stream.
    Keepalive,
}

/// Assemble a completed tool call list from a chunk stream — used by the
/// sampler and engine. Fragments for the same `slot` merge by concat.
///
/// Slots are a map, not a Vec: the Responses API's `output_index` counts
/// **every** output item (reasoning=0, message=1, first call=2…), so call
/// indices are sparse. Gap-filling a Vec materialized an empty-name call
/// per skipped index — the engine then logged "已跳过一个格式异常的空工具
/// 调用" once per reasoning/message item before each real call.
#[derive(Debug, Default)]
pub struct ToolCallAssembler {
    calls: std::collections::BTreeMap<usize, ToolCall>,
}

impl ToolCallAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one chunk. Returns `true` when the chunk was a tool delta.
    pub fn feed(&mut self, chunk: &StreamChunk) -> bool {
        if let StreamChunk::ToolCallDelta {
            slot,
            id,
            name,
            args_delta,
        } = chunk
        {
            let call = self.calls.entry(*slot).or_insert_with(|| ToolCall {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
            });
            if let Some(id) = id {
                // id arrives once — assign, never append (Responses repeats
                // call_id on argument deltas; appending corrupts it).
                if call.id.is_empty() {
                    call.id = id.clone();
                }
            }
            if let Some(name) = name {
                // name likewise arrives once — assign; appending produced
                // `smart_readsmart_read…` when the args delta re-sent it.
                if call.name.is_empty() {
                    call.name = name.clone();
                }
            }
            call.arguments.push_str(args_delta);
            true
        } else {
            false
        }
    }

    /// Completed calls in `slot` order — sparse slots collapse; no
    /// empty slots survive.
    ///
    /// Backfills a synthesized `id` on any call the provider never named:
    /// some Responses-API backends (deepseek) only expose `call_id` on
    /// `output_item.done` — or not at all — and persisting `id: ""`
    /// replays next turn as `function_call.call_id: ""`, which strict
    /// upstreams reject with `400 invalid_request_error`.
    pub fn finish(mut self) -> Vec<ToolCall> {
        static SYNTH_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        for call in self.calls.values_mut() {
            if call.id.is_empty() {
                call.id = format!(
                    "call_{:x}_{}",
                    now_ms(),
                    SYNTH_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                );
            }
            // Harmony `<|…|>` framing leaked into `function.name` breaks
            // dispatch as `unknown tool` — a name never legitimately
            // contains the markers, strip unconditionally.
            call.name = HarmonyStripper::strip_str(&call.name);
            // The same leak inside streamed `arguments` corrupts the JSON
            // (a valid object followed by `argument`/`call` markup prose,
            // or a string cut mid-way by a leaked close marker). Recover
            // before the engine's `_malformed` fallback: strip the tokens,
            // take the first complete JSON value when markup prose trails
            // a valid object, then close a dangling tail. The repair is
            // gated on framing actually being present: a leaked close
            // marker is the MODEL ending the arg, but a bare transport
            // cut's tail is arbitrary — "completing" it could dispatch a
            // call never intended (e.g. `rm -rf /var` cut to `rm -rf /`).
            if serde_json::from_str::<serde_json::Value>(&call.arguments).is_err() {
                let stripped = HarmonyStripper::strip_str(&call.arguments);
                let recovered = serde_json::from_str::<serde_json::Value>(&stripped)
                    .ok()
                    .or_else(|| {
                        serde_json::Deserializer::from_str(&stripped)
                            .into_iter::<serde_json::Value>()
                            .next()
                            .and_then(|r| r.ok())
                    })
                    .or_else(|| {
                        (stripped.len() != call.arguments.len())
                            .then(|| repair_truncated_json(&stripped))
                            .flatten()
                    });
                if let Some(v) = recovered {
                    call.arguments = serde_json::to_string(&v).unwrap_or_default();
                }
            }
        }
        self.calls.into_values().collect()
    }
}

// ---- harmony special-token stripping ----------------------------------------
//
// gpt-oss / harmony-format models emit control tokens as inline text when a
// chat-completions shim surfaces them in `delta.content`/`arguments` instead
// of a structured `tool_calls` block: `<|open|>`, `<|sep|>`, `<|close|>`,
// `<|start|>`, `<|message|>`, `<|channel|>`, `<|end|>`, `<|call|>`,
// `<|return|>`, `<|endoftext|>`, … They are model-internal framing, not user
// text — left in, they corrupt history (the model parrots the marker next
// turn), render as garbage in the UI, and break the accumulated `arguments`
// JSON. One uniform shape covers them all: `<|` + `[a-z0-9_]+` + `|>`.

/// Incremental `<|…|>` token stripper — the chat-completions adapter feeds
/// `content`/`reasoning` deltas through it, and [`ToolCallAssembler::finish`]
/// uses [`HarmonyStripper::strip_str`] on assembled `arguments`/`name`.
///
/// A partial token split across deltas is held until it resolves or proves
/// to be plain text; `flush` at stream end emits whatever was still held.
pub struct HarmonyStripper {
    /// Held-back tail that might still become a `<|…|>` token (`<` / `<|se` …).
    hold: String,
}

impl HarmonyStripper {
    pub fn new() -> Self {
        Self {
            hold: String::new(),
        }
    }

    /// Feed one delta; returns the text safe to emit (None = all held).
    pub fn feed(&mut self, delta: &str) -> Option<String> {
        self.hold.push_str(delta);
        let mut out = String::new();
        let bytes = self.hold.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // A `<` at the buffer tail could still grow into `<|…|>` — hold it.
            if bytes[i] == b'<' && i + 1 == bytes.len() {
                break;
            }
            if bytes[i] == b'<' && bytes.get(i + 1) == Some(&b'|') {
                let rest = &self.hold[i..];
                // Whole token present? `<|word|>` — drop it.
                if let Some(tok_len) = harmony_token_len(rest) {
                    i += tok_len;
                    continue;
                }
                // Could still be a token once more bytes land — the prefix is
                // `<|` plus word chars so far but no closing `|` yet.
                if could_be_harmony_prefix(rest) {
                    break; // hold the tail
                }
                // `<|` followed by a non-word char → literal text, emit both.
                out.push_str("<|");
                i += 2;
                continue;
            }
            let l = utf8_len(bytes[i]);
            out.push_str(&self.hold[i..i + l]);
            i += l;
        }
        self.hold = self.hold[i..].to_string();
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Stream end — anything still held was plain text, not a token. Emit it.
    pub fn flush(&mut self) -> Option<String> {
        if self.hold.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.hold))
        }
    }

    /// One-shot strip for a whole string — assembled `arguments`, `name`.
    pub fn strip_str(s: &str) -> String {
        let mut st = Self::new();
        let mut out = st.feed(s).unwrap_or_default();
        if let Some(tail) = st.flush() {
            out.push_str(&tail);
        }
        out
    }
}

/// If `rest` opens with a complete `<|word|>` harmony token, return its byte
/// length. `word` is `[a-z0-9_]+` — lowercase+digits+underscore only, which is
/// every harmony marker (`<|end|>`, `<|call_0|>`) but not text like `<|Hi|>`.
fn harmony_token_len(rest: &str) -> Option<usize> {
    debug_assert!(rest.starts_with("<|"));
    let body = &rest[2..];
    let close = body.find('|')?;
    let word = &body[..close];
    if !word.is_empty()
        && word
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        // `<|` + word + `|` — but the `>` must follow too.
        if body.as_bytes().get(close + 1) == Some(&b'>') {
            return Some(2 + close + 2); // `<|` + word + `|>`
        }
    }
    None
}

/// True when `rest` is `<|` (or `<|` + word chars) with no closing `|` yet —
/// a token that could complete on the next delta. Also covers a bare `<` or
/// `<|` at the buffer tail.
fn could_be_harmony_prefix(rest: &str) -> bool {
    if rest == "<" || rest == "<|" {
        return true;
    }
    // `<|` + partial word chars, still no `|`/`>` seen → may become a token.
    rest.starts_with("<|")
        && rest[2..]
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// UTF-8 length of the char starting at byte `b`.
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

/// Close a JSON text truncated at a harmony boundary: `"` inside an open
/// string, then `]`/`}` per still-open container. Returns the parsed value
/// only when the repair actually yields valid JSON — a dangling escape or
/// half-literal stays unparseable and the caller keeps the `_malformed`
/// fallback instead.
fn repair_truncated_json(s: &str) -> Option<serde_json::Value> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for ch in s.chars() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    let mut out = s.to_string();
    if in_string {
        out.push('"');
    }
    while let Some(c) = stack.pop() {
        out.push(c);
    }
    serde_json::from_str::<serde_json::Value>(&out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_card_roundtrips_the_frontend_payload() {
        // The card row is persisted verbatim and parsed by the webview —
        // field names are part of the frontend contract.
        let m = ChatMessage::compaction(126_995, 64_019, 77, true, "merged summary");
        assert_eq!(m.role, Role::System);
        assert_eq!(m.notice, Some(NoticeKind::Compacted));
        let payload: serde_json::Value =
            serde_json::from_str(m.content.as_deref().unwrap()).unwrap();
        assert_eq!(payload["before_tokens"], 126_995);
        assert_eq!(payload["after_tokens"], 64_019);
        assert_eq!(payload["removed_messages"], 77);
        assert_eq!(payload["manual"], true);
        assert_eq!(payload["note"], "merged summary");

        // Notice kind serializes to the snake_case string the TS union
        // matches on.
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"compacted\""), "{j}");
    }

    /// The reasoning trace is what a reopened session replays as its
    /// 思考过程 block, so it must survive the JSONL snapshot exactly — and a
    /// trace-free row must not grow the file.
    #[test]
    fn reasoning_roundtrips_and_old_snapshots_default() {
        let m = ChatMessage::assistant("答案").with_reasoning(Some("先想一下".into()));
        let j = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.reasoning.as_deref(), Some("先想一下"));

        // A row written before the field existed still loads.
        let old = r#"{"role":"assistant","content":"hi"}"#;
        let back: ChatMessage = serde_json::from_str(old).unwrap();
        assert_eq!(back.reasoning, None);

        // No trace → no key at all.
        let plain = serde_json::to_string(&ChatMessage::assistant("x")).unwrap();
        assert!(!plain.contains("reasoning"), "{plain}");
    }

    #[test]
    fn opaque_provider_state_roundtrips_and_old_snapshots_stay_compatible() {
        let state = serde_json::json!({
            "origin": "responses-fixture",
            "model": "fixture-gpt",
            "data": "{\"output\":[{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"opaque-fixture\"}]}"
        });
        let message: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant", "content": "answer", "provider_state": state,
        }))
        .unwrap();
        let stored = serde_json::to_value(&message).unwrap();
        assert!(
            !format!("{message:?}").contains("opaque-fixture"),
            "Debug output must not expose opaque replay data"
        );
        assert_eq!(
            stored["provider_state"], state,
            "provider continuity data must survive a session snapshot"
        );
        let old: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":"old"}"#).unwrap();
        assert!(serde_json::to_value(old)
            .unwrap()
            .get("provider_state")
            .is_none());
    }

    #[test]
    fn assembler_collapses_sparse_output_indices() {
        // Responses `output_index` counts reasoning/message items too —
        // a call at index 2 must not materialize empty slots 0 and 1.
        let mut a = ToolCallAssembler::new();
        assert!(a.feed(&StreamChunk::ToolCallDelta {
            slot: 2,
            id: Some("c1".into()),
            name: Some("bash".into()),
            args_delta: "{}".into(),
        }));
        let calls = a.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }

    #[test]
    fn notice_roundtrips_and_old_snapshots_default() {
        // New format serializes `notice`; old snapshots lack it → `None`.
        let m = ChatMessage::notice("已被用户中断");
        let j = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.notice, Some(NoticeKind::System));
        let m = ChatMessage::notice_error("upstream error");
        let j = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.notice, Some(NoticeKind::Error));
        let m = ChatMessage::host_notice("[background task bg-1 finished]");
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"notice\":\"host\""), "{j}");
        let back: ChatMessage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.role, Role::System);
        assert_eq!(back.notice, Some(NoticeKind::Host));
        assert!(!back.notice.unwrap().is_ui_only());
        let old = r#"{"role":"system","content":"a context note"}"#;
        let back: ChatMessage = serde_json::from_str(old).unwrap();
        assert_eq!(back.notice, None);
    }

    /// Card kinds persist as their bare name, so a snapshot written before
    /// cards were generic (`"notice":"plan"`) reads back as the same card.
    #[test]
    fn card_notices_keep_the_bare_kind_on_disk() {
        let m = ChatMessage::card("plan", "{}".into());
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"notice\":\"plan\""), "{j}");
        let old = r#"{"role":"system","content":"{}","notice":"plan"}"#;
        let back: ChatMessage = serde_json::from_str(old).unwrap();
        assert_eq!(back.notice, Some(NoticeKind::Card("plan".into())));
        assert!(back.notice.unwrap().is_ui_only());
        for (name, kind) in NoticeKind::NAMES {
            let j = serde_json::to_string(&kind).unwrap();
            assert_eq!(j, format!("\"{name}\""));
            assert_eq!(serde_json::from_str::<NoticeKind>(&j).unwrap(), kind);
        }
    }

    /// A steer is tagged on disk, still reads as the user's words to the
    /// model, and an old row (no tag, prefix only) is still recognised.
    #[test]
    fn steers_are_tagged_and_old_untagged_rows_still_match() {
        let m = ChatMessage::user_steer("先别改 config");
        assert_eq!(m.notice, Some(NoticeKind::Steer));
        assert_eq!(m.role, Role::User);
        assert_eq!(
            m.content.as_deref(),
            Some("The user interrupted: 先别改 config")
        );
        assert!(m.is_steer());
        assert_eq!(m.steer_text(), Some("先别改 config"));
        assert!(!m.notice.as_ref().unwrap().is_ui_only());

        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains("\"notice\":\"steer\""), "{j}");
        let back: ChatMessage = serde_json::from_str(&j).unwrap();
        assert!(back.is_steer());

        let old = r#"{"role":"user","content":"The user interrupted: use the other file"}"#;
        let back: ChatMessage = serde_json::from_str(old).unwrap();
        assert_eq!(back.notice, None);
        assert!(back.is_steer());
        assert_eq!(back.steer_text(), Some("use the other file"));
    }

    /// Only a user row can be a steer, and a tagged row is judged by its tag:
    /// a hidden instruction or a card that quotes the prefix is not one.
    #[test]
    fn only_untagged_or_steer_user_rows_are_steers() {
        assert!(!ChatMessage::user("do the thing").is_steer());
        assert!(!ChatMessage::user("The user interrupted:no space").is_steer());
        assert!(!ChatMessage::assistant("The user interrupted: x").is_steer());
        assert!(!ChatMessage::user_hidden("The user interrupted: x").is_steer());
        assert!(!ChatMessage::card("plan", "The user interrupted: x".into()).is_steer());
        assert_eq!(ChatMessage::user("hi").steer_text(), None);
    }

    #[test]
    fn empty_content_roundtrips_as_none() {
        // `None` → `""` → `None`, never `Some("")` — the snapshot format
        // must not manufacture a ghost empty string (resume sanitize and
        // every `is_empty` guard treat the two identically, so `""` stays
        // legal on input too).
        let m = ChatMessage {
            role: Role::Assistant,
            content: None,
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            notice: None,
            ts: None,
            images: Vec::new(),
            reasoning: None,
            duration_ms: None,
            provider_state: None,
        };
        let j = serde_json::to_string(&m).unwrap();
        assert!(j.contains(r#""content":"""#), "None must serialize as \"\"");
        let back: ChatMessage = serde_json::from_str(&j).unwrap();
        assert_eq!(back.content, None);
        let back: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":""}"#).unwrap();
        assert_eq!(back.content, None);
    }

    // ---- harmony special-token stripper ----
    //
    // A gpt-oss shim that surfaces harmony framing in `content`/`arguments`
    // must not leak the tokens through. Build the markers by concat so this
    // source file itself never contains a live token literal.
    fn tok(w: &str) -> String {
        format!("{}{}{}", "<|", w, "|>")
    }

    #[test]
    fn strips_full_harmony_tokens() {
        let mut s = HarmonyStripper::new();
        let text = format!("fix {} it {}now{}", tok("open"), tok("sep"), tok("close"));
        assert_eq!(s.feed(&text), Some("fix  it now".to_string()));
    }

    #[test]
    fn strips_token_split_across_deltas() {
        let mut s = HarmonyStripper::new();
        // `<|se` then `p|>` — the partial prefix is held, then the whole token drops.
        assert_eq!(s.feed(&format!("a {}", "<|se")), Some("a ".to_string()));
        assert_eq!(s.feed("p|> tail"), Some(" tail".to_string()));
    }

    #[test]
    fn leaves_literal_angle_pipe_text() {
        let mut s = HarmonyStripper::new();
        // Uppercase after `<|` and a space are NOT harmony-word chars — pass through.
        for src in ["<|Hi|>", "a <| b |> c", "x < y", "<| spaced |>"] {
            assert_eq!(s.feed(src), Some(src.to_string()), "mangled: {src}");
        }
    }

    #[test]
    fn flush_emits_held_non_token_tail() {
        let mut s = HarmonyStripper::new();
        // `feed` emits the safe prefix and holds the trailing lone `<`.
        assert_eq!(s.feed("ends with <"), Some("ends with ".to_string()));
        // …and `flush` at stream end emits just the held `<` tail.
        assert_eq!(s.flush(), Some("<".to_string()));
    }

    #[test]
    fn strips_known_harmony_markers() {
        let mut s = HarmonyStripper::new();
        for w in [
            "open",
            "sep",
            "close",
            "start",
            "message",
            "channel",
            "end",
            "call",
            "return",
            "endoftext",
            "constrain",
        ] {
            let t = tok(w);
            assert_eq!(s.feed(&format!("x{}y", t)), Some("xy".to_string()), "{t}");
        }
    }

    #[test]
    fn strip_str_is_a_one_shot_feed_plus_flush() {
        let s = format!("a{}b{}", tok("open"), tok("close"));
        assert_eq!(HarmonyStripper::strip_str(&s), "ab");
    }

    /// Framing leaked into `arguments` AFTER a complete JSON object — the
    /// markup prose (`argument`/`call` words between tokens) trails the
    /// value. Salvage takes the first complete object: the tool call runs.
    #[test]
    fn assembler_recovers_call_with_harmony_tail_after_json() {
        let mut a = ToolCallAssembler::new();
        a.feed(&StreamChunk::ToolCallDelta {
            slot: 0,
            id: Some("c1".into()),
            name: Some("apply_patch".into()),
            args_delta: format!(
                "{{\"patch\":\"x\"}}{}argument{}{}call{}",
                tok("close"),
                tok("sep"),
                tok("close"),
                tok("sep")
            ),
        });
        let calls = a.finish();
        let v: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(v["patch"], "x");
    }

    /// A leaked close marker ended the arg mid-string — the dangling JSON
    /// tail is repaired so the tool sees real args instead of `_malformed`.
    #[test]
    fn assembler_repairs_args_cut_at_a_harmony_boundary() {
        let mut a = ToolCallAssembler::new();
        a.feed(&StreamChunk::ToolCallDelta {
            slot: 0,
            id: None,
            name: Some("apply_patch".into()),
            args_delta: format!(
                "{{\"patch\":\"*** Begin Patch\\n+unicode: u16{}",
                tok("close")
            ),
        });
        let calls = a.finish();
        let v: serde_json::Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert!(v["patch"].as_str().unwrap().contains("Begin Patch"));
    }

    /// A bare transport cut carries no framing — the tail is arbitrary and
    /// closing it could "complete" a call the model never intended, so it
    /// must stay unparseable (the engine's `_malformed` path reports it).
    #[test]
    fn assembler_leaves_a_bare_transport_cut_malformed() {
        let mut a = ToolCallAssembler::new();
        a.feed(&StreamChunk::ToolCallDelta {
            slot: 0,
            id: None,
            name: Some("bash".into()),
            args_delta: "{\"command\":\"rm -rf /var".into(),
        });
        let calls = a.finish();
        assert!(
            serde_json::from_str::<serde_json::Value>(&calls[0].arguments).is_err(),
            "a bare cut must not be repaired into a runnable command"
        );
    }

    #[test]
    fn assembler_strips_harmony_from_name() {
        let mut a = ToolCallAssembler::new();
        a.feed(&StreamChunk::ToolCallDelta {
            slot: 0,
            id: None,
            name: Some(format!("apply_patch{}", tok("close"))),
            args_delta: "{}".into(),
        });
        assert_eq!(a.finish()[0].name, "apply_patch");
    }

    /// A tool call that legitimately writes `<|…|>` text (a doc/patch about
    /// the markers) keeps its args verbatim — salvage only runs when the
    /// JSON doesn't already parse.
    #[test]
    fn assembler_keeps_valid_args_with_literal_marker_text() {
        let args = format!("{{\"patch\":\"use {} markers\"}}", tok("close"));
        let mut a = ToolCallAssembler::new();
        a.feed(&StreamChunk::ToolCallDelta {
            slot: 0,
            id: None,
            name: Some("apply_patch".into()),
            args_delta: args.clone(),
        });
        assert_eq!(a.finish()[0].arguments, args);
    }
}
