# korei

The agent runtime kernel: a product-neutral async engine that drives an LLM
through tool calls. It knows no workspace, sandbox, desktop, session store or
coding semantics — products and tool packages plug into its seams.

| crate | role |
|---|---|
| `agent-ipc` | UI event / command contract and wire codec |
| `agent-llm-core` | provider contract: `ChatMessage`, `StreamChunk`, `LlmProvider`, `ProviderCompat`, `Sampler`, egress masking, thinking levels |
| `agent-tool` | tool contract: `ToolSpec`, `Commit`, `ToolContext`, `Extensions` |
| `agent-kernel` | engine loop, permissions, compaction, hooks, subagents, todo, `batch_execute` |
| `agent-testkit` | scripted provider for tests (dev-only, not published) |

## Seams

`TurnPolicy`, `PluginRouter`, `SubagentHost`, `TodoStorage`, `BatchProfile`.
`crates/agent-kernel/tests/seam_api.rs` implements every one from outside the
crate; a breaking change to a seam fails there first.

## Laws

- The closure depends on nothing but itself, and its normal dependency tree
  contains no `rig-core`, `reqwest`, `keyring` or `toml`:
  `scripts/check-kernel-deps.sh`.
- `crates/agent-kernel/tests/standalone_assembly.rs` assembles a second product
  from the public API only.

Extensions live in `korei-ext`; provider implementations (`agent-llm`) and
products stay with the product repository.
