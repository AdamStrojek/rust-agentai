# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`agentai` is a Rust library for building LLM agents on top of the [`genai`](https://crates.io/crates/genai) crate. It is pre-1.0 and under heavy development; breaking API changes are acceptable.

Cargo workspace (`resolver = "3"`, members `crates/*`, edition 2021):
- `crates/agentai` – the library.
- `crates/agentai-macros` – proc-macro crate providing `#[toolbox]` (re-exported as `agentai::tool::toolbox`).

Both crates share one version group and are released together by release-plz (`release-plz.toml`) on push to `master`. CHANGELOG.md is generated from commit messages.

## Commands

CI (`.github/workflows/ci.yml`) runs these; keep them passing:

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

- Single test: `cargo test -p agentai <test_name>` (e.g. unit tests in `crates/agentai/src/tool/buildin.rs`).
- Offline tests only: `cargo test -p agentai --lib`. The only unit tests are in `tool/buildin.rs`.
- Doc tests only: `cargo test -p agentai --doc`. Every example is included as a runnable doc test (via `src/examples/*.rs`), so plain `cargo test` executes all examples against a live LLM and fails without the env vars below.
- Example: `cargo run --example <name>` (`simple`, `struct_output`, `tools_custom`, `tools_web`, `tools_mcp`, `genai_custom`, `openrouter`; the last one needs `OPENROUTER_API_KEY`). `scripts/run-examples.sh` runs them all and writes logs to `logs/`.
- Docs: `cargo doc -p agentai --open` (the `examples` and `structured_output` modules exist only under `#[cfg(doc)]`).

Examples and tests that hit an LLM need `AGENTAI_BASE_URL`, `AGENTAI_API_KEY`, and optionally `AGENTAI_MODEL` (CI uses OpenRouter: `https://openrouter.ai/api/v1/`, the free model `nvidia/nemotron-3.5-super-vl:free`, key in the `OPENROUTER_API_KEY` secret). `tools_web` needs `BRAVE_API_KEY`; `tools_mcp` launches `uvx mcp-server-time`, so `uv` must be installed.

## Architecture

**Agent loop** (`crates/agentai/src/agent.rs`): `Agent` holds a genai `Client` and its conversation history as a plain `Vec<ChatMessage>` that starts with the system prompt. Constructors: `Agent::new(system)` uses the default genai client, `new_with_client(client, system)` takes your own, and `new_with_provider(adapter_kind, Option<base_url>, api_key, system)` builds a client bound to one genai adapter (`with_adapter_kind` + `append_provider_config`; `None` keeps the adapter's default endpoint, e.g. native OpenRouter). `new_with_url(base_url, api_key, system)` is `new_with_provider(AdapterKind::OpenAI, Some(base_url), ..)`. `AdapterKind` is re-exported from `agentai::agent`. `new` and `new_with_url` return `Result<_, AgentError>` (`ClientInit`) because genai 0.7 client construction is fallible. `anyhow` is a dev-dependency only; the library uses `thiserror` enums (`AgentError`, `ToolError`). `Agent::run::<D>(model, prompt, Option<&dyn ToolBox>)`:
1. pushes the user prompt into the history;
2. if `D != String`, generates a JSON schema with schemars, strips `$schema`/`title` (Gemini rejects them), and sets it as the response format. This is how structured output works;
3. loops up to `max_iterations` (default `DEFAULT_MAX_ITERATIONS` = 5, set with `with_max_iterations`) and sends the agent's `ChatOptions` (default temperature `DEFAULT_TEMPERATURE` = 0.2, replaced by `with_chat_options`; `run` adds the response format for structured output). Each iteration sends the whole history plus the toolbox definitions to the LLM. If the reply has tool calls (genai 0.7 keeps them next to any text), `into_assistant_message_for_tool_use()` is stored and every call gets a `ToolResponse::from_tool_call`; recoverable tool errors (`ToolError::is_recoverable()`: `NotFound`, `InvalidArguments`, `Failed`), and calls made without a toolbox, are sent back to the LLM as text, while a fatal tool error (`Configuration`, `Fatal`) stops the run with `AgentError::Tool`. Running out of iterations is `AgentError::MaxIterationsReached`, an unparsable answer is `AgentError::InvalidAnswer { raw, .. }`, and `AgentError::is_transient()` flags retryable provider failures. Otherwise all text parts are joined, stored and deserialized into `D` (a `String` result is JSON-quoted first so that `serde_json::from_str` works).

History is kept across `run` calls on the same `Agent`, so reusing an agent continues the conversation. Configurable memory lives on the unmerged `feature/memory` branch.

**Tools** (`tool/mod.rs`): the `ToolBox` trait (`#[async_trait]`, `Send + Sync`) has `tools_definitions()` and `call_tool(name, Value)`. `ToolBoxSet` combines several toolboxes and dispatches to the first one that does not return `ToolError::NotFound`, so that variant is the "try the next toolbox" signal. Feature-gated implementations: `buildin` (`tools-buildin`), `web` (`tools-web`, Brave search plus URL fetch), and `mcp` (`mcp-client`, `McpToolBox` on the official `rmcp` SDK: stdio via `new`, Streamable HTTP via `new_http`; unknown tool names return `NotFound`, server-reported tool errors become `ToolError::Failed`, a broken transport is `ToolError::Fatal`). All features are on by default.

**`#[toolbox]` macro** (`agentai-macros/src/lib.rs`): applied to an `impl` block. Every method marked `#[tool]` (or `#[tool(name = "...")]`) must take `&self` and return `ToolResult`. For each tool the macro generates a hidden `<Method>Params` struct deriving serde and `JsonSchema` (draft 2020-12, no meta-schema), with parameter doc comments carried over as field descriptions. It then generates a `ToolBox` impl whose `call_tool` uses a `match` on the tool name; argument deserialization errors become `ToolError::InvalidArguments` and errors returned by the method pass through unchanged. The generated code refers to `Tool`, `ToolBox`, `ToolError`, `ToolResult`, `serde`, `serde_json`, `schemars`, and `async_trait` by unqualified or crate paths, so the caller's crate must have them in scope or as dependencies.

## Documentation conventions

- `README.md` between `<!-- cargo-rdme start/end -->` is generated from the crate docs in `crates/agentai/src/lib.rs` by cargo-rdme. Edit `lib.rs` and then regenerate; do not edit that section by hand.
- Feature-flag docs come from the `## ` / `#!` comments in `crates/agentai/Cargo.toml` (via `document-features`).
- `llms.txt` (index, llmstxt.org format) and `llms-full.txt` (complete usage guide) in the repo root are hand-written summaries of the public API for LLMs. When a change touches the public API, feature flags, tool names or `Agent::run` behaviour, update both files and the version they name.
- Each example in `crates/agentai/examples/X.rs` has a matching docs-only module `crates/agentai/src/examples/X.rs` that `include_str!`s it. When you add an example, add that module and register it in `src/examples/mod.rs`.

## Commits

Use the `<type>(<scope>): <message>` format (CONTRIBUTING.md). release-plz groups changelog entries by prefix: `feat`, `changed`, `fix`, `deprecated`, and everything else under Miscellaneous.
