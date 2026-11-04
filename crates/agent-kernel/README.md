# agent-kernel

The agent loop dispatches tools and manages context, hooks and subagents.
Providers, tools and host policy enter through the contracts exported by
[`src/lib.rs`](src/lib.rs).

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
