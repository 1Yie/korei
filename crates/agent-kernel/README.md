# agent-kernel

The agent loop dispatches tools and manages context, hooks and subagents.
Providers, tools and host policy enter through the contracts exported by
[`src/lib.rs`](src/lib.rs).

## Execution strategies

`Engine` defaults to `EngineStrategy::ReAct`. A host selects
`EngineStrategy::CodeMode` with `set_strategy` and installs one
`Arc<dyn CodeModeHandler>` with `set_code_mode_handler`. The public contract
lives in [`code_mode`](src/code_mode.rs). Both strategies use the same turn
loop and neutral provider tool-call/result protocol. Code Mode exposes only
the `execute_code` schema; the registry is used for actual capabilities, never
to resolve the control handler. A response mixing control and direct capability
calls is rejected as a whole. Missing handlers and invalid requests fail closed.

The handler borrows `CodeModeContext` for this execution: `Engine`, `EngineIo`,
the current `InvocationTurn`, original control call, trusted outer request ID,
tool context, optional parent identity/deadline and remaining call allowance.
It creates scoped gateways through `context.engine.invocation_gateway` using
`context.io` and `context.turn`. Code, JSON params and optional requested limits
are untrusted inputs validated against the existing bounded JSON/schema
boundary. The host clamps requested limits and binds code/params to its VM.
Identity and scope authority come from trusted host state.

`max_tool_rounds` counts assistant tool rounds in either strategy. The separate
`set_max_code_mode_tool_calls(Some(limit))` bounds actual inner gateway attempts
across all programs and rounds of a Code Mode turn; its default is `None`.
`remaining_tool_calls == usize::MAX` represents no configured parent call cap.
Every host program still requires finite runtime and dispatch limits. The host
must reserve from the remaining allowance before each dispatch, atomically for
concurrent calls, and return actual `attempted_calls`. Program/scope creation
cannot reset parent or policy budgets. The kernel accounts returned attempts
and forces a final tool-free sample when a configured cap is exhausted. A host
report exceeding the allowance is a contract error: subsequent programs stop,
and the error explicitly preserves uncertainty about effects already incurred.

`CodeModeOutcome` contains only an explicit model projection, error flag,
actual attempt count and optional trusted `InvocationStopReason`. The kernel
bounds projection text to `MAX_MODEL_RESULT_BYTES` with an explicit truncation
marker and correlates one outer tool result to the original call. Raw inner
outputs are never automatically appended to history. Partial failures belong
in the projection and its error flag. `UserRejected { request_id }` must come
from gateway audit provenance. Human refusal and execution deadline stop the
remaining prewritten controls in that assistant response and return to the
model. Cancellation stops the turn. Ordinary policy errors and guest error
strings carry no human-refusal authority.

The kernel owns outer queued/started/finished correlation, including rejection
and unavailable-handler paths. The outer control's execution state is UI-only:
it invokes neither root state hooks nor root tool hooks. Actual inner calls
run hooks through the host's scoped invocation gateway. Input, reasoning and
response lifecycle hooks retain their normal behavior. A host session mirror
must suppress hook dispatch for outer control UI events or rebind it to the
execution scope after durable execution begin; kernel UI delivery alone does
not enforce that host boundary. Typed `agent_ipc::code_mode` payloads use bounded
`UiEvent::Custom` envelopes named `code_mode.started`, `code_mode.progress` and
`code_mode.finished`. The host supplies execution/session/agent identity and
optional inner request correlation. Emit start before progress. The host owns
the execution ledger, admission stop, cancellation, gateway completion drain,
artifact finalization and terminal audit; only after these complete may it emit
its unique execution finished event and return. The kernel awaits handler
cleanup cooperatively and emits outer finished after return. Console/summary
carry bounded diagnostics and opaque artifact references, never full raw tool
outputs. The kernel implements this contract and channel, not a VM or sandbox.

## Invocation gateway

[`invocation`](src/invocation.rs) exposes a turn-borrowed gateway for discovery,
authorization, execution and audit. Hosts create an `InvocationTurn` through
`Engine::begin_invocation_turn` and obtain handles with
`Engine::invocation_gateway`. Handles for one turn share policy, mutation and
approval serialization. Discovery follows the live mode and registry; dispatch
rechecks tool identity and permissions after asynchronous authorization work.
ReAct and nested batch calls use this same invocation path. Child agents invoke
tools under their own engine permissions.

[`batch_execute`](src/tools/batch.rs) accepts its canonical calls envelope and
the parser's alternate envelopes, bare arrays and single-call forms. Its shared
schema checks the root structure; malformed items fail individually. Item
aliases, inline arguments and host-declared signature inference normalize before
nested gateway dispatch, which retains argument validation, observation-only
eligibility, permissions, audit and resource limits.

The gateway validates arguments against bounded local JSON schemas, including
arguments changed by hooks. Reviews, staged preparation, scope grants and
publication retain their authorization boundaries. Critical operations require
human approval; a scope grant can trigger a separate tool approval when current
policy requires it.

Hosts requiring durable audit provide an `AuditSink`. Its begin and pre-effect
records must succeed before the associated operation proceeds. Phase and
authorization fields distinguish preparation from an approved write. Audit
waits are bounded; finish has a separate cleanup budget so cancellation can still
settle the invocation. A finish error preserves the indication that effects may
already have occurred. The host owns persistence and reconciliation after an
indeterminate write. An invocation without a sink provides outcome metadata
without durable audit storage.

`InvocationOutcome` preserves the complete tool result and media. Its audit
preview is bounded separately; hosts retain raw results before preparing a
model-facing projection. `InvocationAudit::stop_reason` distinguishes a negative
human verdict from policy denial, approval timeout, invocation deadline and
cancellation. Hosts detect explicit refusal through `UserRejected` and its
request ID. The `denied` flag is a broader ReAct turn-suppression hint.

The gateway regression suite covers these boundaries:

```sh
nice -n 10 cargo test -p agent-kernel --test invocation_gateway --test approval_timeout --test cancel_running_tool
```

## Execution scopes

[`InvocationScope`](src/invocation_scope.rs) owns the services and lifetime of
one host execution. The host supplies its `ToolCtx`, `InvocationIdentity`,
`HookChain`, optional plugin handle, optional `AuditSink` and deadline, then
borrows it through `InvocationGateway::with_scope`. Scope creation leaves the
existing `InvocationTurn` and policy counters intact. Engine permissions and
the live registry remain authoritative; scope services cannot replace them.

The gateway binds review, preparation, policy, grant, dispatch, commit and tool
hooks to that context. Scoped discovery uses the supplied plugin handle; an
absent scoped router exposes no root plugins. Execution, agent, session, trace
and parent identity come from the Rust binding rather than request metadata.
Nested calls preserve their gateway-generated parent invocation identity.
The scope's audit sink owns both parent and child records; install it on the
scope rather than supplying a separate gateway sink.

`ScopeGrant` callbacks return a future. The gateway awaits the callback under
the invocation's cancellation and deadline checks, after required pre-effect
audit and live authority checks. It rechecks tool identity and permissions
before retrying. Preparation carries no write authorization. Hosts must keep
effects inside their returned futures and retain the grant's access boundary.

Scoped tool and state hooks run inline with the current `ToolCtx`. Cancellation
and deadlines drop their pending futures; after-tool cleanup and audit finish
have bounded waits. `HookChain::observe_nowait` skips scoped contexts. Hosts
mirroring `StateChanged` must use the active scoped context or suppress their
root session mirror while that execution owns state publication;
`is_invocation_scoped` identifies such contexts. The ordinary ReAct state pump
retains its session behavior.

`SubagentSpawner::with_scoped_services` is required for scoped delegation. Its
factory receives the current parent context and a fresh child context and
returns `ScopedChildServices`: child hooks, an optional plugin handle, execution
strategy, optional Code Mode handler and optional Code Mode call cap. Code Mode
requires a handler before the child starts; ReAct callers select `ReAct` with no
handler or Code Mode cap. The cap uses the engine's existing turn allowance,
with `None` preserving its default. The factory installs the child's
execution-bound host capabilities explicitly; inheritable parent extensions
remain available in the fresh child context. `with_policy` likewise
receives both contexts and installs the child's own product controllers. Core
keeps the child's own registry, permission gate and headless restrictions;
parent bypass and session approvals do not become child authority. Child audit
records inherit execution/session/trace identity and name their own agent and
parent invocation. Child answering and waiting are restricted to the execution
that created the child.

Call `InvocationScope::close().await` before releasing execution resources. It
rejects new invocations and children, cancels existing work and waits for
invocations, background children, forwarding and detached outcome delivery.
Child cancellation uses a child-local flag, with a bounded cooperative cleanup
window; scope cancellation leaves the global Agent/session cancel unchanged.
Close has a bounded wait and returns an error while work remains owned: retain
the host resources and retry cleanup in that case. Dropping the scope requests
cancellation; awaited close establishes drain completion.

### Host migration

- Await `ToolCtx::grant_scope`; change `install_scope_grant` callbacks to return
  `BoxFuture` and keep asynchronous effects inside that future.
- Add the current `ToolCtx` parameter to `TurnPolicy::before_dispatch` and
  tool/state `AgentHook` methods. `TurnPolicy::after_execute` observes completed
  effects under a bounded wait; its cancellation preserves a completed result.
  Adapt private workspace tracking from the supplied context.
- Change `SubagentSpawner::with_policy` factories to accept parent and child,
  and install `with_scoped_services` for scoped delegation. Build sandbox,
  workspace and lifecycle services from the supplied current parent rather
  than a captured root context. Filter any child router to the child's intended
  capabilities; the child permission gate still authorizes each dispatch.
- Bind the scoped audit sink and trusted identity when constructing the scope,
  route or suppress the session state mirror, and await close before releasing
  runtime/sandbox leases. Core supplies these contracts; host-specific service
  factories, persistence and mirrors are host integration work.

## Tool invocation events

[`UiEvent`](../agent-ipc/src/events.rs) correlates queued, started and finished
tool events with `call_id`. The engine announces a round before dispatch, and a
queued call can finish without starting when cancellation ends the round.
Consumers match invocation identity rather than tool name. Nested batch calls
and child cards carry their own identities, and child forwarding preserves the
identity and parent scope.

## Shared task boards

The [`todo` tool](src/tools/todo.rs) supports pending, in-progress and completed
items, progress notes and one level of children. Tool requests address displayed
board labels, including the parent reference when creating a child.

Hosts supply opaque-row persistence through `TodoStorage` and install it with
`install_shared`. Delegated contexts inherit the same `TodoStorageSlot`; writes
to one board are serialized by its storage identity. A context without installed
storage uses an in-memory board for its own lifetime.

## Background subagents

[`delegate`](src/tools/delegate.rs) and `answer_child` accept `background: true`
to return while the child continues. Child outcomes reach the host's
`ChildNotifier`; `active_child_infos` includes background status. Background mode
and parallel `tasks` are mutually exclusive.

When no independent work remains, the parent calls `wait_child` to keep its
tool round open until the next handoff. The host connects `ChildNotifier` to
that round's notice receiver: waiting returns delivery status, while the notice
carries the question or report once.
Hosts retain the session's `SubagentSpawner` across turns so background handoffs
remain addressable.

A host installs `BackgroundCancel` through `install_background_cancel` so child
lifetime can outlast a turn without inheriting its resettable cancellation flag.
The host owns session cancellation and shutdown through the child-control APIs.
