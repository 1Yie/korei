# korei

A small, product-neutral runtime for LLM agents: it drives a model through tool
calls and knows nothing about your workspace, sandbox or UI. You bring the
tools and the rules; korei runs the loop.

## Crates

| crate | what it is |
|---|---|
| `agent-kernel` | the agent loop: tool dispatch, permissions, context compaction, hooks, subagents |
| `agent-llm-core` | the provider contract: messages, streaming chunks, `LlmProvider`, retry and sampling |
| `agent-tool` | the tool contract: `ToolSpec`, `ToolContext`, `Commit` |
| `agent-ipc` | the event and command types a frontend talks to the kernel with |
| `agent-testkit` | a scripted provider for tests |

## Extending it

The kernel exposes a few seams instead of features: `TurnPolicy`,
`PluginRouter`, `SubagentHost`, `TodoStorage` and `BatchProfile`. Provider
implementations, tool packages and sandboxing live outside the kernel; see
[korei-ext](https://github.com/1Yie/korei-ext) for ready-made tools.

## License

MIT
