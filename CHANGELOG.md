# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.2.0...agentai-v0.3.0) - 2026-10-22

### Breaking changes

See [MIGRATION.md](https://github.com/AdamStrojek/rust-agentai/blob/master/MIGRATION.md#020--030-2026-10-22) for the steps to migrate from 0.2.0, with before and after code.

- `Agent::run(model, prompt, toolbox)` is replaced by `agent.run(prompt).await`, which returns `AgentResult<D>` instead of `D`; the model and toolbox are set with `Agent::with_model` and `Agent::with_toolbox`, or per run.
- The agent no longer remembers earlier runs: pass the history with `with_history(&history)` and append `result.new_messages`. The system prompt is not part of the history.
- An answer cut off by the token limit is `AgentError::MaxTokens` instead of `AgentError::InvalidAnswer`; `AgentError::Llm` and `AgentError::ClientInit` hold an `LlmError`.
- `Agent::new_with_provider` takes a `Provider` instead of `AdapterKind`; `Agent::new_with_client` is removed.
- `ToolChoice` and `ReasoningEffort` are imported from `agentai`.

### New features
- *(agent)* [**breaking**] streaming, caller-owned history, run details and agentai's own types ([#66](https://github.com/AdamStrojek/rust-agentai/pull/66)) (by @AdamStrojek)

### Miscellaneous
- reorder What's New for 0.2.0 by what matters to users ([#64](https://github.com/AdamStrojek/rust-agentai/pull/64)) (by @AdamStrojek) - #64
- update Future Plans and fix the README logo on crates.io ([#63](https://github.com/AdamStrojek/rust-agentai/pull/63)) (by @AdamStrojek) - #63

### Contributors

* @AdamStrojek

## [0.2.0](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.1.5...agentai-v0.2.0) - 2026-10-20

### Breaking changes

See also [MIGRATION.md](https://github.com/AdamStrojek/rust-agentai/blob/master/MIGRATION.md#015--020-2026-10-20) for before and after code.

Steps to migrate from 0.1.x:

- **Toolchain and dependencies**
  - Minimum Rust version is 1.88, and the crates use edition 2024.
  - `genai` 0.3 → 0.7 (also re-exported as `agentai::genai`). Code that builds its own genai client must follow genai's migration guides; for example, `Client::builder().build()` now returns `Result` and `Client::default()` is gone.
  - Answer types and tool parameter types must derive `JsonSchema` from `schemars` 1.x (was 0.9).
  - `reqwest` 0.13 with rustls by default (was native-tls).
- **`Agent` constructors and `run`**
  - `Agent::new` and `Agent::new_with_url` return `Result<Agent, AgentError>`. Add `?`.
  - `Agent::run` returns `Result<D, AgentError>` instead of `anyhow::Result<D>`. `?` inside an `anyhow` function keeps working. To handle errors, match on `AgentError` (`ClientInit`, `Llm`, `Schema`, `InvalidAnswer`, `MaxIterationsReached`, `Tool`), and use `is_transient()` to decide on retries.
  - With `new_with_url`, model names with a foreign genai namespace (for example `anthropic::...`) now fail with an adapter mismatch error. Plain names such as `vendor/model:free` work as before.
- **Tools**
  - `agentai::tool::Tool` is genai 0.7's `Tool`: `name` is a `ToolName` (compare with `.as_str()`), and the struct has more fields. Build it with `Tool::new(name).with_description(..).with_schema(..)` instead of a struct literal.
  - `ToolError` variants:
    - `NoToolFound` is renamed to `NotFound`;
    - `LLMError` is replaced by `InvalidArguments` or `Failed`;
    - `ExecutionError`, `ToolsDefinitionNotReady` and `Other(anyhow::Error)` are removed. Use `Configuration`, `Fatal`, or `ToolError::fatal(err)`.
  - `ToolError` is `#[non_exhaustive]`, and there is no `From<anyhow::Error>` any more. Map errors inside tool methods to a `ToolError`.
  - Fatal tool errors (`Configuration`, `Fatal`) now stop `Agent::run` with `AgentError::Tool`. Recoverable errors (`NotFound`, `InvalidArguments`, `Failed`) are still sent to the model.
  - `#[toolbox]` no longer replaces errors with `ExecutionError`. Errors returned by a tool reach the model as they are, and invalid arguments become `InvalidArguments`.
  - `McpToolBox::call_tool` returns `NotFound` for tools the server does not have, and `Failed` for errors the server reports (`isError`). Before, it returned `Ok` with the error text.

### New features
- *(macros)* add #[tool(strict)] for strict tool schemas ([#61](https://github.com/AdamStrojek/rust-agentai/pull/61)) (by @AdamStrojek)
- *(agent)* configurable max_iterations and ChatOptions ([#59](https://github.com/AdamStrojek/rust-agentai/pull/59)) (by @AdamStrojek)
- *(agent)* add Agent::new_with_provider and native OpenRouter support ([#58](https://github.com/AdamStrojek/rust-agentai/pull/58)) (by @AdamStrojek)
- *(error)* [**breaking**] replace anyhow with typed AgentError and ToolError ([#56](https://github.com/AdamStrojek/rust-agentai/pull/56)) (by @AdamStrojek)
- *(mcp)* replace mcp_client_rs with the official rmcp SDK ([#53](https://github.com/AdamStrojek/rust-agentai/pull/53)) (by @AdamStrojek)
- [**breaking**] upgrade genai to 0.7, schemars 1, reqwest 0.13 and edition 2024 ([#52](https://github.com/AdamStrojek/rust-agentai/pull/52)) (by @AdamStrojek) - #52
- mark Send and Sync on Toolbox trait ([#32](https://github.com/AdamStrojek/rust-agentai/pull/32)) (by @fvanzee) - #32

### Fixes
- *(tools)* replace deprecated time format parsing in buildin tools ([#41](https://github.com/AdamStrojek/rust-agentai/pull/41)) (by @AdamStrojek)

### Miscellaneous
- add project logo ([#54](https://github.com/AdamStrojek/rust-agentai/pull/54)) (by @AdamStrojek) - #54
- run LLM tests against a free OpenRouter model ([#43](https://github.com/AdamStrojek/rust-agentai/pull/43)) (by @AdamStrojek) - #43
- *(deps)* upgrade agentai-macros to syn 3 ([#60](https://github.com/AdamStrojek/rust-agentai/pull/60)) (by @AdamStrojek)

### Contributors

* @AdamStrojek
* @fvanzee

## [0.1.5](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.1.4...agentai-v0.1.5) - 2025-07-20

### New features
- ToolBox ([#18](https://github.com/AdamStrojek/rust-agentai/pull/18)) (by @AdamStrojek) - #18

### Miscellaneous
- Chore/Prepare to release version 0.1.5 ([#27](https://github.com/AdamStrojek/rust-agentai/pull/27)) (by @AdamStrojek) - #27

### Contributors

* @AdamStrojek
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4](https://github.com/AdamStrojek/rust-agentai/compare/v0.1.3...v0.1.4) - 2025-05-21

### Changed

- Hide LLM client ([#16](https://github.com/AdamStrojek/rust-agentai/pull/16))
- Feature/MCP error handling ([#12](https://github.com/AdamStrojek/rust-agentai/pull/12))

## [0.1.3](https://github.com/AdamStrojek/rust-agentai/compare/v0.1.2...v0.1.3) - 2025-01-14

### Added
- Feature/MCP Support ([#9](https://github.com/AdamStrojek/rust-agentai/pull/9))

### Other

- Chore/MCP Client tools and documentation ([#10](https://github.com/AdamStrojek/rust-agentai/pull/10))
- Chore/CI ([#8](https://github.com/AdamStrojek/rust-agentai/pull/8))

## [0.1.2](https://github.com/AdamStrojek/rust-agentai/compare/v0.1.1...v0.1.2) - 2025-01-05

### Changed

- Fix links in README
- Update documentation (#7)
- *(example)* Custom Tool Example (#6)
- Implement Web Search Tool + add examples
- update documentation for Agent structure
- clippy updates
- convert tabs into spaces

## [0.1.1](https://github.com/AdamStrojek/rust-agentai/compare/v0.1.0...v0.1.1) - 2024-12-29

### Added

- *(example)* New example for structured output
- Add documentation for examples, update simple example

### Changed

- Move AgentTool to separate module

### Fixed

- Fix behaviour when return String type

## 0.1.0

### Added

- Initial release
- Simple example demonstrating functionality
