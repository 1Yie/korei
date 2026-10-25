//! `SamplerActor` — resilience policy wrapped around any `LlmProvider`.
//!
//! | Policy         | Value                                                        |
//! |----------------|--------------------------------------------------------------|
//! | Idle timeout   | no chunk for 300 s → `IdleTimeout` error                     |
//! | Doom-loop      | identical repeated generation → abort + resample, ≤3 retries |
//! | Retry          | exponential backoff on transport errors + 429/5xx, never 4xx |
//! | Mid-stream cut | retry resends the same messages → provider regenerates;      |
//! |                | the caller drops attempt-scoped buffers on `Retrying`        |
//! | Usage          | the last `Done` chunk's counts feed `SessionStats`            |
//!
//! The sampler never owns a concrete provider — it wraps `Arc<dyn LlmProvider>`,
//! so a hot-swap replaces the Arc, not the actor.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use thiserror::Error;

use crate::provider::LlmProvider;
use crate::provider::ModelParams;
use crate::types::{ChatMessage, StreamChunk};

/// No chunk for this long → the stream is wedged; retry.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
/// Stream *establishment* ceiling — `provider.chat_stream` itself is awaited
/// before `stream.next()` ever runs, so `IDLE_TIMEOUT` never covers a hang in
/// the request setup (DNS resolve, TLS handshake, pool checkout). A stuck
/// setup otherwise wedges the turn forever with zero visible feedback.
pub const ESTABLISH_TIMEOUT: Duration = Duration::from_secs(60);
/// Identical generation repeated across this many recent BYTES → model is
/// stuck. Byte-bounded, not chunk-bounded: providers chunk deltas anywhere
/// from 1 char to whole paragraphs, so a fixed chunk count made the analysis
/// window (and its period bounds) a function of the backend's segmentation.
pub const DOOM_WINDOW_BYTES: usize = 512;
/// Retry ceiling for doom loops and transport errors.
pub const MAX_RETRIES: u32 = 3;
/// Backoff base for transport/429/5xx retries.
pub const RETRY_BASE: Duration = Duration::from_millis(500);

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("stream idle for {}s", IDLE_TIMEOUT.as_secs())]
    IdleTimeout,

    #[error("doom loop detected: identical generation repeated")]
    DoomLoop,

    #[error("provider error after {attempts} attempts: {message}")]
    Exhausted { attempts: u32, message: String },

    #[error("stream error: {0}")]
    Stream(String),
}

/// How often the sampler calls `on_idle` while it waits for the next chunk.
/// Short enough that a coalescing buffer holding already-received text can be
/// flushed promptly when the stream goes quiet.
pub const IDLE_TICK: Duration = Duration::from_millis(40);

/// Events the sampler emits *besides* model chunks — surfaced to the UI as
/// `system` messages so degrade is visible, never silent.
#[derive(Debug, Clone)]
pub enum SamplerEvent {
    /// Retry k of N scheduled after `delay`.
    Retrying {
        attempt: u32,
        delay: Duration,
        reason: String,
    },
    /// Provider failed permanently; the session should degrade or abort.
    Failed(String),
    /// The primary exhausted its retries and the chain moved to the next
    /// provider. The partial attempt is discarded: the new provider answers
    /// the request from the start.
    FellBack { from: String, to: String },
}

/// Stateless resilience wrapper around one provider at a time.
/// `set_provider` hot-swaps for the *next* call — in-flight streams finish.
pub struct Sampler {
    provider: Arc<dyn LlmProvider>,
    /// Ordered degrade chain tried after `provider` exhausts its retries.
    /// Each entry is `(config name, provider)`; empty means no fallback.
    fallbacks: Vec<(String, Arc<dyn LlmProvider>)>,
    /// Config name of the primary, for `FellBack` events.
    primary_name: String,
    /// Egress masker — scrubs resolved secrets from outbound request bodies.
    masker: crate::masking::EgressMasker,
}

#[derive(Debug, Clone, Copy)]
pub struct SampleRequest<'a> {
    pub model: &'a str,
    pub temperature: f32,
    pub tools: Option<&'a serde_json::Value>,
    pub reasoning_effort: Option<&'a str>,
    /// Per-model wire settings (`maxTokens`, `samplingParams`, model-level
    /// `compat`) — resolved from config by the caller, forwarded to the
    /// adapter. Use [`ModelParams::EMPTY`] when the model declares nothing.
    pub params: &'a ModelParams,
}

impl Sampler {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            fallbacks: Vec::new(),
            primary_name: String::new(),
            masker: crate::masking::EgressMasker::new(vec![]),
        }
    }

    /// Install the degrade chain. Tried in order after the primary exhausts
    /// its retries on a provider-side failure; a request rejected with a 4xx
    /// is never sent down the chain, because another vendor would reject it
    /// too.
    pub fn set_fallbacks(&mut self, chain: Vec<(String, Arc<dyn LlmProvider>)>) {
        self.fallbacks = chain;
    }

    /// Name the primary provider, so a `FellBack` event can say where it came from.
    pub fn set_primary_name(&mut self, name: impl Into<String>) {
        self.primary_name = name.into();
    }

    /// Install the resolved secrets for egress masking — called once config
    /// resolves `env:`/`keyring:` handles into real key material.
    pub fn set_masker(&mut self, secrets: Vec<String>) {
        self.masker = crate::masking::EgressMasker::new(secrets);
    }

    /// Hot-swap: next call uses the new provider. In-flight calls unaffected.
    pub fn set_provider(&mut self, provider: Arc<dyn LlmProvider>) {
        self.provider = provider;
    }

    pub fn provider(&self) -> Arc<dyn LlmProvider> {
        self.provider.clone()
    }

    /// One full request with retry/doom-loop policy. Calls `on_chunk` per
    /// chunk and `on_event` for policy events; returns the assembled text.
    ///
    /// `on_idle` is called every [`IDLE_TICK`] while the stream is quiet. A
    /// caller that coalesces chunks (the engine batches text for the UI)
    /// flushes its buffer there, so what the model already sent is not
    /// stranded until the next chunk — or a cancel — shakes it loose.
    ///
    /// Chunk ordering guarantee: `Done` reaches `on_chunk` exactly once per
    /// attempt, so the caller treats `Err` from this function as the attempt
    /// boundary, not a missing `Done`.
    pub async fn sample<F, G, I>(
        &self,
        req: SampleRequest<'_>,
        messages: &[ChatMessage],
        mut on_chunk: F,
        mut on_event: G,
        mut on_idle: I,
    ) -> Result<(), SampleError>
    where
        F: FnMut(&StreamChunk),
        G: FnMut(&SamplerEvent),
        I: FnMut(),
    {
        // Primary first, then the degrade chain in config order. A provider
        // that exhausts its retries hands the request to the next one; a
        // permanent error (4xx, parse failure) stops the walk, because another
        // vendor would reject the same request.
        let mut current = (self.primary_name.clone(), self.provider.clone());
        let mut hop = 0usize;
        loop {
            match self
                .sample_with(
                    &current.1,
                    &req,
                    messages,
                    &mut on_chunk,
                    &mut on_event,
                    &mut on_idle,
                )
                .await
            {
                Err(SampleError::Exhausted { .. }) if hop < self.fallbacks.len() => {
                    let next = self.fallbacks[hop].clone();
                    hop += 1;
                    on_event(&SamplerEvent::FellBack {
                        from: current.0.clone(),
                        to: next.0.clone(),
                    });
                    current = next;
                }
                other => return other,
            }
        }
    }

    /// The retry policy against one provider: transient failures back off and
    /// retry the same request, up to [`MAX_RETRIES`] attempts.
    async fn sample_with<F, G, I>(
        &self,
        provider: &Arc<dyn LlmProvider>,
        req: &SampleRequest<'_>,
        messages: &[ChatMessage],
        on_chunk: &mut F,
        on_event: &mut G,
        on_idle: &mut I,
    ) -> Result<(), SampleError>
    where
        F: FnMut(&StreamChunk),
        G: FnMut(&SamplerEvent),
        I: FnMut(),
    {
        let mut attempt = 0u32;
        loop {
            match self
                .sample_once(provider, req, messages, on_chunk, on_idle)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) if !e.retryable() => return Err(e),
                Err(e) => {
                    attempt += 1;
                    if attempt >= MAX_RETRIES {
                        on_event(&SamplerEvent::Failed(e.to_string()));
                        return Err(SampleError::Exhausted {
                            attempts: attempt,
                            message: e.to_string(),
                        });
                    }
                    let delay = RETRY_BASE * 2u32.saturating_pow(attempt - 1);
                    on_event(&SamplerEvent::Retrying {
                        attempt,
                        delay,
                        reason: e.to_string(),
                    });
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// One attempt: drive the provider stream under idle-timeout and
    /// doom-loop supervision. **Egress masking** scrubs secret material out
    /// of the request messages before they hit the wire; **stream salvage**
    /// returns the partial text alongside the error so the caller can keep
    /// it visible + retry with a continuation prompt.
    async fn sample_once<F, I>(
        &self,
        provider: &Arc<dyn LlmProvider>,
        req: &SampleRequest<'_>,
        messages: &[ChatMessage],
        on_chunk: &mut F,
        on_idle: &mut I,
    ) -> Result<(), SampleError>
    where
        F: FnMut(&StreamChunk),
        I: FnMut(),
    {
        // Egress mask: scrub secrets from outbound messages (defense in
        // depth — sandbox env sanitize is the first wall, this is second).
        // Scrub BOTH `content` and `tool_calls[].arguments` — a tool that
        // read a secret (`.env`, `cat ~/.aws/credentials`) puts it into the
        // assistant's arguments on the NEXT request, which is the most
        // common exfiltration path and was previously unmasked.
        let masked: Vec<ChatMessage> = messages
            .iter()
            .map(|m| {
                let mut m = m.clone();
                if let Some(c) = &m.content {
                    let (s, _) = self.masker.scrub(c);
                    m.content = Some(s);
                }
                if let Some(calls) = &mut m.tool_calls {
                    for call in calls.iter_mut() {
                        let (s, _) = self.masker.scrub(&call.arguments);
                        call.arguments = s;
                    }
                }
                m
            })
            .collect();

        let mut stream = tokio::time::timeout(
            ESTABLISH_TIMEOUT,
            provider.chat_stream(
                req.model,
                &masked,
                req.tools.cloned(),
                req.temperature,
                req.reasoning_effort,
                req.params,
            ),
        )
        .await
        .map_err(|_| {
            SampleError::Stream(format!(
                "provider did not begin streaming within {}s",
                ESTABLISH_TIMEOUT.as_secs()
            ))
        })?
        .map_err(|e| classify_transport_error(&e.to_string()))?;

        let mut recent: VecDeque<String> = VecDeque::with_capacity(64);
        let mut recent_bytes = 0usize;
        let mut done_seen = false;

        loop {
            // Wait for the next chunk in short ticks so the caller can flush
            // text it is still holding while the stream is quiet. The idle
            // ceiling counts from the last sign of life: content, or a
            // `Keepalive` — an upstream that is silently thinking.
            let mut idle_since = tokio::time::Instant::now();
            let chunk = loop {
                let remaining = IDLE_TIMEOUT.saturating_sub(idle_since.elapsed());
                if remaining.is_zero() {
                    return Err(SampleError::IdleTimeout);
                }
                match tokio::time::timeout(IDLE_TICK.min(remaining), stream.next()).await {
                    Err(_) => on_idle(),
                    Ok(None) => break None,
                    Ok(Some(Err(e))) => {
                        let msg = e.to_string();
                        let err = classify_transport_error(&msg);
                        return Err(err);
                    }
                    Ok(Some(Ok(StreamChunk::Keepalive))) => {
                        idle_since = tokio::time::Instant::now();
                    }
                    Ok(Some(Ok(c))) => break Some(c),
                }
            };
            let Some(chunk) = chunk else { break };

            if let StreamChunk::Done { .. } = &chunk {
                if done_seen {
                    continue; // provider double-fired Done; swallow
                }
                done_seen = true;
            }

            // Doom-loop: fold content deltas into a sliding window and check
            // for a repeated suffix period.
            // All three text-bearing variants count — a reasoning model can
            // doom-loop inside its thinking trace, or inside streamed tool
            // arguments, without a single ContentDelta ever arriving.
            let doom_text = match &chunk {
                StreamChunk::ContentDelta(t) | StreamChunk::ReasoningDelta(t) => Some(t.as_str()),
                StreamChunk::ToolCallDelta { args_delta, .. } => Some(args_delta.as_str()),
                _ => None,
            };
            if let Some(t) = doom_text {
                recent_bytes += t.len();
                recent.push_back(t.to_string());
                // Detect BEFORE trimming — the just-pushed chunk is what fills
                // the window past DOOM_WINDOW_BYTES; evicting first would
                // leave `joined` under the gate and the check would never run.
                if doom_detected(&recent) {
                    return Err(SampleError::DoomLoop);
                }
                while recent_bytes > DOOM_WINDOW_BYTES && recent.len() > 1 {
                    if let Some(old) = recent.pop_front() {
                        recent_bytes -= old.len();
                    }
                }
            }

            // A provider-level error INSIDE the stream (devin emits
            // `response.failed`/`error` events → `StreamChunk::Error`, e.g.
            // "internal_server_error … upstream error") is a real failure —
            // escalate to Err so the retry loop sees it. Forward the chunk
            // first so the UI still shows the message.
            if let StreamChunk::Error(msg) = &chunk {
                on_chunk(&chunk);
                return Err(classify_transport_error(msg));
            }

            on_chunk(&chunk);
        }

        if !done_seen {
            // Clean close without Done — synthesize so the caller's
            // invariants hold (spec: Done fires exactly once).
            on_chunk(&StreamChunk::Done {
                prompt_tokens: None,
                completion_tokens: None,
                cached_tokens: None,
            });
        }
        Ok(())
    }
}

/// Sliding-window doom detector: true when the window's tail is a strict
/// repetition of some period ≥4 chars. Catches "abababab…" and verbatim
/// paragraph loops alike without needing cryptographic comparison.
fn doom_detected(recent: &VecDeque<String>) -> bool {
    let joined: String = recent.iter().cloned().collect();
    let b = joined.as_bytes();
    let n = b.len();
    // Judge only a full window — the same bytes-per-window contract the old
    // 64-chunk gate approximated.
    if n < DOOM_WINDOW_BYTES {
        return false;
    }
    // Three equal tail slices alone isn't a doom loop — markdown lists,
    // repeated `className=` props, and `- ` bullets legitimately produce
    // identical 3× tails in a 64-char window on NORMAL output. A real doom
    // loop is the whole window being ~periodic: ≥85% of positions satisfy
    // b[i] == b[i-p] AND the tail shows ≥3 clean repeats. The 85% slack
    // tolerates the window boundary landing mid-repetition.
    for p in 4..=(n / 3) {
        // Tail must show ≥3 verbatim repeats first — the cheap check.
        if b[n - p..] != b[n - 2 * p..n - p] || b[n - 2 * p..n - p] != b[n - 3 * p..n - 2 * p] {
            continue;
        }
        let mut matches = 0usize;
        for i in p..n {
            if b[i] == b[i - p] {
                matches += 1;
            }
        }
        // matches/(n-p) ≥ 85% → the window is periodic with p.
        if matches * 20 >= (n - p) * 17 {
            return true;
        }
    }
    false
}

/// Decide retryability from an error string: 429/5xx and transport-level
/// failures retry; other 4xx never do (spec: "never on other 4xx").
fn classify_transport_error(msg: &str) -> SampleError {
    // `send_sse` prefixes real HTTP failures with `provider HTTP <code>` — a
    // parsed status beats substring guessing. Without it, a body that merely
    // CONTAINS a number ("max_tokens 15000", a request id with "500" in it)
    // was retried as if it were a 5xx.
    let code = msg
        .strip_prefix("provider HTTP ")
        .and_then(|s| s.split([':', ' ']).next())
        .and_then(|s| s.parse::<u16>().ok());
    let retryable = match code {
        Some(c) => c == 429 || c >= 500,
        // Fallback for failures that never carried a status — stream-chunk
        // errors, transport drops, gateway prose. Keyword-only on purpose:
        // bare digits in a free-text body are not a status code.
        None => {
            let lower = msg.to_lowercase();
            lower.contains("timeout")
                || lower.contains("timed out")
                || lower.contains("connection")
                || lower.contains("transport")
                || lower.contains("eof")
                || lower.contains("reset")
                // Gateway-style 5xx bodies that don't carry the numeric code —
                // devin reports `internal_server_error … upstream error`.
                || lower.contains("internal_server_error")
                || lower.contains("internal server error")
                || lower.contains("upstream error")
                || lower.contains("bad gateway")
                || lower.contains("service unavailable")
                || lower.contains("rate limit")
                || lower.contains("http 429")
                || lower.contains("http 5")
        }
    };
    if retryable {
        SampleError::Stream(format!("retryable: {msg}"))
    } else {
        SampleError::Stream(format!("fatal: {msg}"))
    }
}

/// Human-facing text for policy events — the engine sends these to the UI as
/// `SystemMessage` and persists them into history, so Debug output
/// (`Retrying { attempt: 1, delay: 500ms, … }`) is not acceptable there.
impl std::fmt::Display for SamplerEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerEvent::Retrying {
                attempt,
                delay,
                reason,
            } => write!(
                f,
                "上游响应中断 — {:.1}s 后重试（第 {attempt}/{MAX_RETRIES} 次）：{reason}",
                delay.as_secs_f32()
            ),
            SamplerEvent::Failed(reason) => write!(f, "上游持续不可用：{reason}"),
            SamplerEvent::FellBack { from, to } => {
                write!(f, "提供商 {from} 不可用，已切换到备用提供商 {to}")
            }
        }
    }
}

impl SampleError {
    /// Whether the sampler may retry this failure.
    fn retryable(&self) -> bool {
        match self {
            SampleError::IdleTimeout | SampleError::DoomLoop => true,
            SampleError::Stream(s) => s.starts_with("retryable:"),
            SampleError::Exhausted { .. } => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 400 whose body contains "500" is not a 5xx — the structured
    /// `provider HTTP <code>` prefix must beat body substring guessing,
    /// else a permanent failure burns the whole retry budget with backoff.
    #[test]
    fn classify_prefers_the_real_status_over_body_text() {
        // The reported false positive: a token-limit message containing 500.
        let e = classify_transport_error(
            "provider HTTP 400: {\"error\":\"max_tokens 15000 exceeds context\"}",
        );
        assert!(!e.retryable(), "a 400 must never retry: {e}");

        for code in [429u16, 500, 502, 503, 504] {
            let e = classify_transport_error(&format!("provider HTTP {code}: x"));
            assert!(e.retryable(), "HTTP {code} should retry");
        }
        for code in [400u16, 401, 403, 404, 422] {
            let e = classify_transport_error(&format!("provider HTTP {code}: x"));
            assert!(!e.retryable(), "HTTP {code} must not retry");
        }

        // No prefix → keyword fallback (in-stream errors, transport drops).
        assert!(classify_transport_error("internal_server_error: upstream error").retryable());
        assert!(classify_transport_error("connection reset by peer").retryable());
        // A bare number in free text is still not a status code.
        assert!(!classify_transport_error("model refused: 5000 tokens over budget").retryable());
    }

    /// The doom window is byte-bounded: providers chunk deltas anywhere from
    /// one char to whole paragraphs, so the detector must see the same ~512B
    /// regardless of segmentation.
    #[test]
    fn doom_window_is_byte_bounded() {
        // One fat chunk can fill the window on its own.
        let mut fat = VecDeque::new();
        fat.push_back("x".repeat(DOOM_WINDOW_BYTES + 64));
        assert!(doom_detected(&fat));

        // Many tiny chunks of the same repeated pattern.
        let mut many = VecDeque::new();
        for _ in 0..256 {
            many.push_back("ab".to_string());
        }
        assert!(doom_detected(&many));

        // A short window never fires.
        let mut small = VecDeque::new();
        small.push_back("ab".repeat(30));
        assert!(!doom_detected(&small));
    }

    // ---- stream-level retry/resume coverage -----------------------------
    //
    // `sample` is the only place transport errors turn into retries; a
    // scripted provider makes the whole attempt stream observable: how many
    // `chat_stream` calls happened, what `on_chunk`/`on_event` saw, and
    // which error class ended the run.

    use crate::provider::BoxStream;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// One scripted attempt: either `chat_stream` itself fails (establish
    /// phase) or the stream yields these items then closes.
    enum Script {
        EstablishFail(&'static str),
        Chunks(Vec<anyhow::Result<StreamChunk>>),
    }

    struct ScriptedProvider {
        calls: AtomicUsize,
        scripts: Mutex<VecDeque<Script>>,
        /// Message contents as they reached the wire, per call.
        seen: Mutex<Vec<Vec<String>>>,
    }

    impl ScriptedProvider {
        fn of(scripts: Vec<Script>) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                scripts: Mutex::new(scripts.into()),
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl LlmProvider for ScriptedProvider {
        fn id(&self) -> &'static str {
            "scripted"
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
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.seen
                .lock()
                .unwrap()
                .push(_messages.iter().filter_map(|m| m.content.clone()).collect());
            let script = self
                .scripts
                .lock()
                .unwrap()
                .pop_front()
                .expect("sampler called chat_stream more times than scripted");
            match script {
                Script::EstablishFail(msg) => Err(anyhow::anyhow!(msg.to_string())),
                Script::Chunks(items) => Ok(Box::pin(futures::stream::iter(items))),
            }
        }
    }

    fn req<'a>() -> SampleRequest<'a> {
        SampleRequest {
            model: "m",
            temperature: 0.0,
            tools: None,
            reasoning_effort: None,
            params: &ModelParams::EMPTY,
        }
    }

    fn delta(t: &str) -> anyhow::Result<StreamChunk> {
        Ok(StreamChunk::ContentDelta(t.to_string()))
    }

    fn done() -> anyhow::Result<StreamChunk> {
        Ok(StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        })
    }

    fn stream_err(msg: &str) -> anyhow::Result<StreamChunk> {
        Err(anyhow::anyhow!(msg.to_string()))
    }

    /// A model that thinks silently for longer than the idle ceiling keeps the
    /// stream alive with `Keepalive`; the sampler must not call it wedged, and
    /// must not hand the marker on as if it were output.
    #[tokio::test(start_paused = true)]
    async fn keepalive_outlasts_the_idle_ceiling_and_is_not_forwarded() {
        let pings = (IDLE_TIMEOUT.as_secs() / 60 + 2) as usize;
        let slow = futures::stream::unfold(0usize, move |n| async move {
            if n < pings {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Some((Ok(StreamChunk::Keepalive), n + 1))
            } else if n == pings {
                Some((Ok(StreamChunk::ContentDelta("answer".into())), n + 1))
            } else if n == pings + 1 {
                Some((Ok(done_chunk()), n + 1))
            } else {
                None
            }
        });
        let p = Arc::new(StreamingProvider(std::sync::Mutex::new(Some(Box::pin(
            slow,
        )))));
        let sampler = Sampler::new(p);

        let mut seen: Vec<StreamChunk> = Vec::new();
        let started = tokio::time::Instant::now();
        sampler
            .sample(req(), &[], |c| seen.push(c.clone()), |_| {}, || {})
            .await
            .expect("a silent model with a live connection must not time out");

        assert!(
            started.elapsed() > IDLE_TIMEOUT,
            "the test must outlast the ceiling"
        );
        assert!(!seen.contains(&StreamChunk::Keepalive), "{seen:?}");
        assert!(
            matches!(&seen[0], StreamChunk::ContentDelta(t) if t == "answer"),
            "{seen:?}"
        );
        assert!(
            matches!(seen.last(), Some(StreamChunk::Done { .. })),
            "{seen:?}"
        );
    }

    /// No sign of life at all still trips the ceiling.
    #[tokio::test(start_paused = true)]
    async fn silence_without_keepalive_still_times_out() {
        let dead = futures::stream::pending::<anyhow::Result<StreamChunk>>();
        let p = Arc::new(StreamingProvider(std::sync::Mutex::new(Some(Box::pin(
            dead,
        )))));
        let sampler = Sampler::new(p);

        let err = sampler
            .sample(req(), &[], |_| {}, |_| {}, || {})
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                SampleError::IdleTimeout | SampleError::Exhausted { .. }
            ),
            "{err:?}"
        );
    }

    fn done_chunk() -> StreamChunk {
        StreamChunk::Done {
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
        }
    }

    /// Hands out one prepared stream; any retry after it gets a silent one.
    struct StreamingProvider(std::sync::Mutex<Option<BoxStream<StreamChunk>>>);

    #[async_trait]
    impl LlmProvider for StreamingProvider {
        fn id(&self) -> &'static str {
            "streaming"
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
            let prepared = self.0.lock().unwrap().take();
            Ok(prepared.unwrap_or_else(|| Box::pin(futures::stream::pending())))
        }
    }

    /// The resolved key must never reach the provider in clear text, even when
    /// the user pastes it into a prompt. Without `set_masker` the wire carries
    /// the raw key, which is the exfiltration path this masker exists to close.
    #[tokio::test]
    async fn resolved_secret_is_scrubbed_before_it_reaches_the_wire() {
        let secret = "sk-live-abcdef1234567890";
        let p = ScriptedProvider::of(vec![Script::Chunks(vec![done()])]);
        let mut sampler = Sampler::new(p.clone());
        sampler.set_masker(vec![secret.to_string()]);

        let messages = vec![ChatMessage::user(format!("my key is {secret}, use it"))];
        sampler
            .sample(req(), &messages, |_| {}, |_| {}, || {})
            .await
            .expect("sample");

        let seen = p.seen.lock().unwrap();
        let first = seen.first().expect("one call");
        assert!(
            first.iter().all(|c| !c.contains(secret)),
            "plaintext key reached the provider: {first:?}"
        );
    }

    /// A primary that exhausts its retries falls back to the next provider in
    /// the chain, and the switch is announced so it is never silent.
    #[tokio::test]
    async fn exhausted_primary_falls_back_to_next_provider() {
        let primary = ScriptedProvider::of(vec![
            Script::EstablishFail("provider HTTP 503: a"),
            Script::EstablishFail("provider HTTP 503: b"),
            Script::EstablishFail("provider HTTP 503: c"),
        ]);
        let backup = ScriptedProvider::of(vec![Script::Chunks(vec![delta("ok"), done()])]);
        let mut sampler = Sampler::new(primary.clone());
        sampler.set_fallbacks(vec![("backup".into(), backup.clone())]);

        let mut chunks: Vec<String> = Vec::new();
        let mut events: Vec<String> = Vec::new();
        sampler
            .sample(
                req(),
                &[],
                |c| {
                    if let StreamChunk::ContentDelta(t) = c {
                        chunks.push(t.clone());
                    }
                },
                |e| events.push(format!("{e:?}")),
                || {},
            )
            .await
            .expect("the backup must answer");

        assert_eq!(chunks, vec!["ok"]);
        assert_eq!(backup.calls.load(Ordering::SeqCst), 1);
        assert!(
            events
                .iter()
                .any(|e| e.contains("FellBack") && e.contains("backup")),
            "{events:?}"
        );
    }

    /// A 4xx is the request's fault, not the provider's: another vendor would
    /// reject the same request, so the chain must NOT be walked.
    #[tokio::test]
    async fn client_error_does_not_fall_back() {
        let primary = ScriptedProvider::of(vec![Script::EstablishFail(
            "provider HTTP 400: max_tokens exceeds context",
        )]);
        let backup = ScriptedProvider::of(vec![Script::Chunks(vec![done()])]);
        let mut sampler = Sampler::new(primary.clone());
        sampler.set_fallbacks(vec![("backup".into(), backup.clone())]);

        let err = sampler
            .sample(req(), &[], |_| {}, |_| {}, || {})
            .await
            .unwrap_err();
        assert!(err.to_string().contains("400"), "{err}");
        assert_eq!(
            backup.calls.load(Ordering::SeqCst),
            0,
            "4xx must not fall back"
        );
    }

    /// When every provider in the chain fails, the error names the last one
    /// tried, so the user knows the chain itself is exhausted.
    #[tokio::test]
    async fn exhausted_chain_reports_the_last_failure() {
        let primary = ScriptedProvider::of(vec![
            Script::EstablishFail("provider HTTP 503: a"),
            Script::EstablishFail("provider HTTP 503: b"),
            Script::EstablishFail("provider HTTP 503: c"),
        ]);
        let backup = ScriptedProvider::of(vec![
            Script::EstablishFail("provider HTTP 502: x"),
            Script::EstablishFail("provider HTTP 502: y"),
            Script::EstablishFail("provider HTTP 502: z"),
        ]);
        let mut sampler = Sampler::new(primary);
        sampler.set_fallbacks(vec![("backup".into(), backup)]);

        let err = sampler
            .sample(req(), &[], |_| {}, |_| {}, || {})
            .await
            .unwrap_err();
        assert!(err.to_string().contains("502"), "{err}");
    }

    /// A retryable mid-stream cut (HTTP 5xx class) retries the SAME request:
    /// attempt 2 re-establishes, deltas replay, Done still arrives once.
    #[tokio::test]
    async fn midstream_transport_cut_retries_and_recovers() {
        let p = ScriptedProvider::of(vec![
            Script::Chunks(vec![delta("hel"), stream_err("provider HTTP 503: down")]),
            Script::Chunks(vec![delta("hel"), delta("lo"), done()]),
        ]);
        let sampler = Sampler::new(p.clone());

        let mut chunks: Vec<String> = Vec::new();
        let mut events: Vec<String> = Vec::new();
        sampler
            .sample(
                req(),
                &[],
                |c| {
                    if let StreamChunk::ContentDelta(t) = c {
                        chunks.push(t.clone());
                    }
                },
                |e| events.push(format!("{e:?}")),
                || {},
            )
            .await
            .expect("retry must recover");

        assert_eq!(
            p.calls.load(Ordering::SeqCst),
            2,
            "one retry, one extra call"
        );
        // The retry re-streams from scratch — the caller sees "hel" again.
        assert_eq!(chunks, vec!["hel", "hel", "lo"]);
        assert!(
            events.iter().any(|e| e.contains("Retrying")),
            "a Retrying event must be reported, got: {events:?}"
        );
    }

    /// A non-retryable error (4xx) fails on the FIRST attempt — no retry,
    /// no Retrying event, exactly one provider call.
    #[tokio::test]
    async fn non_retryable_error_fails_without_retry() {
        let p = ScriptedProvider::of(vec![Script::Chunks(vec![stream_err(
            "provider HTTP 401: invalid key",
        )])]);
        let sampler = Sampler::new(p.clone());

        let mut events: Vec<String> = Vec::new();
        let err = sampler
            .sample(req(), &[], |_| {}, |e| events.push(format!("{e:?}")), || {})
            .await
            .unwrap_err();

        assert_eq!(p.calls.load(Ordering::SeqCst), 1, "401 must not retry");
        assert!(!events.iter().any(|e| e.contains("Retrying")));
        assert!(err.to_string().contains("401"));
    }

    /// Retryable establish-phase failure: `chat_stream` itself returning
    /// Err classifies and retries like a mid-stream cut.
    #[tokio::test]
    async fn establish_phase_error_retries() {
        let p = ScriptedProvider::of(vec![
            Script::EstablishFail("provider HTTP 502: bad gateway"),
            Script::Chunks(vec![done()]),
        ]);
        let sampler = Sampler::new(p.clone());
        sampler
            .sample(req(), &[], |_| {}, |_| {}, || {})
            .await
            .expect("establish retry recovers");
        assert_eq!(p.calls.load(Ordering::SeqCst), 2);
    }

    /// Every attempt a retryable failure → the budget burns to
    /// `Exhausted { attempts: MAX_RETRIES }` and a `Failed` event lands.
    #[tokio::test]
    async fn retryable_failure_exhausts_budget() {
        let p = ScriptedProvider::of(vec![
            Script::Chunks(vec![stream_err("provider HTTP 503: a")]),
            Script::Chunks(vec![stream_err("provider HTTP 503: b")]),
            Script::Chunks(vec![stream_err("provider HTTP 503: c")]),
        ]);
        let sampler = Sampler::new(p.clone());

        let mut events: Vec<String> = Vec::new();
        let err = sampler
            .sample(req(), &[], |_| {}, |e| events.push(format!("{e:?}")), || {})
            .await
            .unwrap_err();

        match err {
            SampleError::Exhausted { attempts, .. } => {
                assert_eq!(attempts, MAX_RETRIES);
            }
            other => panic!("expected Exhausted, got {other:?}"),
        }
        assert_eq!(
            p.calls.load(Ordering::SeqCst),
            MAX_RETRIES as usize,
            "exactly MAX_RETRIES provider calls"
        );
        assert!(events.iter().any(|e| e.contains("Failed")));
    }

    /// `Done` ordering guarantee: a provider that double-fires is swallowed
    /// to one; a clean close without `Done` synthesizes one — either way
    /// the caller's invariant is exactly-one-Done per attempt.
    #[tokio::test]
    async fn done_fires_exactly_once_per_attempt() {
        // Double Done → one reaches on_chunk.
        let p = ScriptedProvider::of(vec![Script::Chunks(vec![delta("x"), done(), done()])]);
        let sampler = Sampler::new(p);
        let mut dones = 0usize;
        sampler
            .sample(
                req(),
                &[],
                |c| {
                    if matches!(c, StreamChunk::Done { .. }) {
                        dones += 1;
                    }
                },
                |_| {},
                || {},
            )
            .await
            .unwrap();
        assert_eq!(dones, 1, "provider double-Done is swallowed");

        // Clean close, no Done → synthesized.
        let p = ScriptedProvider::of(vec![Script::Chunks(vec![delta("x")])]);
        let sampler = Sampler::new(p);
        let mut dones = 0usize;
        sampler
            .sample(
                req(),
                &[],
                |c| {
                    if matches!(c, StreamChunk::Done { .. }) {
                        dones += 1;
                    }
                },
                |_| {},
                || {},
            )
            .await
            .unwrap();
        assert_eq!(dones, 1, "missing Done is synthesized");
    }

    /// An in-stream `StreamChunk::Error` (provider `response.failed` event)
    /// is forwarded to on_chunk AND treated as a real failure: the retry
    /// loop re-establishes instead of silently finishing.
    #[tokio::test]
    async fn in_stream_error_chunk_forwards_then_retries() {
        let p = ScriptedProvider::of(vec![
            Script::Chunks(vec![
                delta("a"),
                Ok(StreamChunk::Error(
                    "internal_server_error: upstream error".into(),
                )),
            ]),
            Script::Chunks(vec![done()]),
        ]);
        let sampler = Sampler::new(p.clone());

        let mut saw_error_chunk = false;
        sampler
            .sample(
                req(),
                &[],
                |c| {
                    if matches!(c, StreamChunk::Error(_)) {
                        saw_error_chunk = true;
                    }
                },
                |_| {},
                || {},
            )
            .await
            .expect("error chunk is retryable and recovers");

        assert!(saw_error_chunk, "the error chunk must reach on_chunk");
        assert_eq!(p.calls.load(Ordering::SeqCst), 2);
    }
}
