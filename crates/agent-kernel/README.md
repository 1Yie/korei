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
