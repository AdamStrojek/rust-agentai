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

- Single test: `cargo test -p agentai <test_name>` (e.g. unit tests in `crates/agentai/src/agent/tests.rs`).
- Offline tests only: `cargo test -p agentai --lib` (agent loop against a mock LLM in `agent/tests.rs`, message conversion in `message.rs`, plus `tool/buildin.rs` and `tool/mcp.rs`; the MCP tests need `uvx`, and `test_get_location` calls the live Nominatim API).
- Doc tests only: `cargo test -p agentai --doc`. Every example is included as a runnable doc test (via `src/examples/*.rs`), so plain `cargo test` executes all examples against a live LLM and fails without the env vars below.
- Example: `cargo run --example <name>` (`simple`, `streaming`, `event_listener`, `struct_output`, `tools_custom`, `tools_web`, `tools_mcp`, `vision`, `openrouter`; the last one needs `OPENROUTER_API_KEY` and is only compiled by `cargo test`). `scripts/run-examples.sh` runs them all and writes logs to `logs/`.
- Docs: `cargo doc -p agentai --open` (the `examples` and `structured_output` modules exist only under `#[cfg(doc)]`).

Examples and tests that hit an LLM read only `AGENTAI_MODEL` (default `claude-haiku-5-5`), a genai model name: the agent is built with `Agent::new`, so the provider comes from the name and its key and endpoint from genai's environment variables (`ANTHROPIC_API_KEY`; locally `omlx::Qwen3.8-27B-4bit` with `OMLX_ENDPOINT` and `OMLX_API_KEY`). CI uses `claude-haiku-5-5` with the `ANTHROPIC_API_KEY` secret. `tools_web` needs `BRAVE_API_KEY`; `tools_mcp` launches `uvx mcp-server-time`, so `uv` must be installed; `vision` needs a model that accepts images.

## Architecture

**Public types are our own** (no genai types in the public API, `genai` is an implementation detail): `message.rs` (`Message { role, parts }`, `Part`, `ToolCall`, `ToolResponse`, `Media`, `ProviderData`; own serde format, lossless `From` conversions to/from `genai::chat::ChatMessage` tested in its unit tests), `agent/model.rs` (`ChatOptions` + `to_genai()`, `ToolChoice`, `ReasoningEffort`, `Usage`, `StopReason`, `Provider` mapped to `AdapterKind`), `tool::Tool` (converted to `genai::chat::Tool`), `LlmError` (wraps `genai::Error` as a boxed source with `status()`/`is_transient()`). When genai adds something users need, add our own field/variant (`#[non_exhaustive]`) and map it. Naming rule: every setter is `with_*`, getters are bare names.

**Agent** (`crates/agentai/src/agent/mod.rs`): immutable, cheap-to-clone configuration: genai `Client` (private), system prompt, optional default model (`with_model`) and toolbox (`with_toolbox`, stored as `Arc<dyn ToolBox>`), `max_iterations` (default `DEFAULT_MAX_ITERATIONS` = 5) and `ChatOptions` (default temperature `DEFAULT_TEMPERATURE` = 0.2). It does not keep a conversation. Constructors: `Agent::new(system)` (provider from the model name), `new_with_provider(Provider, Option<base_url>, api_key, system)` (`with_adapter_kind` + `append_provider_config`; `None` keeps the default endpoint) and `new_with_url(base_url, api_key, system)` = `new_with_provider(Provider::OpenAI, Some(base_url), ..)`. All return `Result<_, AgentError>` (`ClientInit`). There is no way to pass a custom genai client. `anyhow` is a dev-dependency only; the library uses `thiserror` enums.

**Runs** (`request.rs`, `stream.rs`): `Agent::run::<D>(prompt)` (`prompt: impl IntoPrompt`, see `prompt.rs`: `&str`/`String` via a blanket impl over `Into<String>`, `Media`, or a `Prompt` of text and media parts that becomes the first user message) and `Agent::run_stream::<D>(prompt)` return the same `AgentRequest<'a, D, M>` builder (`with_history(&[Message])`, `with_model/with_toolbox/with_max_iterations/with_chat_options`, `with_options(RunOptions)`); the mode marker `M` (`Complete`/`Streaming`) only selects the `IntoFuture` impl. Awaiting `Complete` gives `AgentResult<D>` (`answer`, `new_messages`, summed `usage`, `stop_reason`, `reasoning_content`, `iterations`); awaiting `Streaming` validates the config and gives `AgentStream<D>` (`Send + 'static`) of `AgentEvent`s (`TextDelta`, `ReasoningDelta`, `ToolCall`, `ToolResult`, `Finished(AgentResult)`); `AgentStream::listen`/`listen_async` pass every event to an `EventListener`/`AsyncEventListener` (`listener.rs`; listeners get a borrowed, answer-type-free `RunEvent` so they stay dyn-compatible and can later be attached to `Agent`, for example for a GUI). `listener::Printer` is the ready-made listener, configured on itself with `with_*`; `print_to` (`std::io::Write`), `print_to_async` (`futures::io::AsyncWrite`) and `print_to_tokio` (feature `tokio`, via `Printer::tokio`) are shortcuts that build a default `Printer`. The stream itself takes no printing options. Keep the core runtime-neutral (the goal is to build for wasm later): tokio-specific code goes behind the `tokio` feature. The caller owns the history: it is only read, and `AgentResult::new_messages` holds just this run's messages. The system prompt is sent with `ChatRequest::with_system` and never stored in the history.

**Agent loop** (`core.rs`, `run_loop`): one `async_stream::try_stream!` used by both modes (`streaming: bool`; awaiting `run` drains it with `finish`). The response format is resolved up front in `AgentRequest::resolve` (if `D != String`, a schemars JSON schema with `$schema`/`title` stripped, since Gemini rejects them). The loop runs up to `max_iterations`, each time sending system prompt + history + this run's messages (converted to genai) + toolbox definitions. Non-streaming uses `exec_chat`; streaming uses `exec_chat_stream` with all `capture_*` options on, yields deltas and rebuilds the step from `StreamEnd`. If the reply has tool calls, `into_assistant_message_for_tool_use()` is stored (converted to `Message`) and every call gets a `ToolResponse`; recoverable tool errors (`ToolError::is_recoverable()`: `NotFound`, `InvalidArguments`, `Failed`), and calls made without a toolbox, are sent back to the LLM as text, while a fatal tool error (`Configuration`, `Fatal`) stops the run with `AgentError::Tool`. Otherwise all text parts are joined and deserialized into `D` (a `String` result is JSON-quoted first so that `serde_json::from_str` works). Errors: `MissingModel`, `MaxTokens { raw }` (checked right after every model call, before tools run), `MaxIterationsReached`, `InvalidAnswer { raw, .. }`, `Llm(LlmError)`; `AgentError::is_transient()` flags retryable provider failures. Unit tests in `agent/tests.rs` use a mock OpenAI-compatible server (JSON and SSE replies).

**Tools** (`tool/mod.rs`): the `ToolBox` trait (`#[async_trait]`, `Send + Sync`) has `tools_definitions()` and `call_tool(name, Value)`. `ToolBoxSet` combines several toolboxes and dispatches to the first one that does not return `ToolError::NotFound`, so that variant is the "try the next toolbox" signal. Feature-gated implementations: `buildin` (`tools-buildin`), `web` (`tools-web`, Brave search plus URL fetch), and `mcp` (`mcp-client`, `McpToolBox` on the official `rmcp` SDK: stdio via `new`, Streamable HTTP via `new_http`; unknown tool names return `NotFound`, server-reported tool errors become `ToolError::Failed`, a broken transport is `ToolError::Fatal`). All features are on by default.

**`#[toolbox]` macro** (`agentai-macros/src/lib.rs`): applied to an `impl` block. Every method marked `#[tool]` (or `#[tool(name = "...")]`) must take `&self` and return `ToolResult`. For each tool the macro generates a hidden `<Method>Params` struct deriving serde and `JsonSchema` (draft 2020-12, no meta-schema), with parameter doc comments carried over as field descriptions. It then generates a `ToolBox` impl whose `call_tool` uses a `match` on the tool name; argument deserialization errors become `ToolError::InvalidArguments` and errors returned by the method pass through unchanged. The generated code refers to `Tool`, `ToolBox`, `ToolError`, `ToolResult`, `serde`, `serde_json`, `schemars`, and `async_trait` by unqualified or crate paths, so the caller's crate must have them in scope or as dependencies.

## Documentation conventions

- Describe changes from the library user's point of view: what they can now do, or must do differently, when building agents. This applies to "What's New", changelog notes, PR descriptions and `llms*.txt`. Order items by how much they change the user's code or what they can build. Put internal work last and keep it short, for example dependency upgrades, refactors and CI. For a dependency upgrade, name the user-visible effect (for example "tool calls are no longer lost"), not just the version.
- "What's New" entries (in `lib.rs`, mirrored in the README) use the heading format `#### X.Y.Z - <highlights>`. If the release has breaking changes, end the entry with a short **Breaking changes** paragraph that names them and tells the user to read the CHANGELOG before upgrading.
- `MIGRATION.md` in the repo root is the migration guide. Every release with breaking changes adds a section at the top, `## X.Y.Z → A.B.C (YYYY-MM-DD)` from the previous release to the new one (one jump only, never from older releases), with the date of the new release and before/after code for each change users must make. Mention only what user code actually touches. The release's CHANGELOG entry and the "What's New" Breaking changes paragraph link to its section.
- `README.md` between `<!-- cargo-rdme start/end -->` is generated from the crate docs in `crates/agentai/src/lib.rs` by cargo-rdme. Edit `lib.rs` and then regenerate; do not edit that section by hand.
- Feature-flag docs come from the `## ` / `#!` comments in `crates/agentai/Cargo.toml` (via `document-features`).
- `llms.txt` (index, llmstxt.org format) and `llms-full.txt` (complete usage guide) in the repo root are hand-written summaries of the public API for LLMs. When a change touches the public API, feature flags, tool names or run behaviour, update both files and the version they name.
- Each example in `crates/agentai/examples/X.rs` has a matching docs-only module `crates/agentai/src/examples/X.rs` that `include_str!`s it. When you add an example, add that module and register it in `src/examples/mod.rs`.

## Commits

Use the `<type>(<scope>): <message>` format (CONTRIBUTING.md). release-plz groups changelog entries by prefix: `feat`, `changed`, `fix`, `deprecated`, and everything else under Miscellaneous.
