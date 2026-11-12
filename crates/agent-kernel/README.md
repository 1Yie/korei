# agent-kernel

The agent loop dispatches tools and manages context, hooks and subagents.
Providers, tools and host policy enter through the contracts exported by
[`src/lib.rs`](src/lib.rs).

## Invocation gateway

[`invocation`](src/invocation.rs) exposes a turn-borrowed gateway for discovery,
authorization, execution and audit. Hosts create an `InvocationTurn` through
`Engine::begin_invocation_turn` and obtain handles with
`Engine::invocation_gateway`. Handles for one turn share policy, mutation and
approval serialization. Discovery follows the live mode and registry; dispatch
rechecks tool identity and permissions after asynchronous authorization work.
ReAct and nested batch calls use this same invocation path. Child agents invoke
tools under their own engine permissions.

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
returns child hooks and an optional child plugin handle. It must install the
child's execution-bound host capabilities explicitly. `with_policy` likewise
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
