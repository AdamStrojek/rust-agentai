<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AdamStrojek/rust-agentai/master/assets/logo-dark.png">
  <img align="right" src="https://raw.githubusercontent.com/AdamStrojek/rust-agentai/master/assets/logo-docs.png" alt="AgentAI logo" width="120">
</picture>

[![](https://img.shields.io/crates/v/agentai.svg)](https://crates.io/crates/agentai)
[![](https://docs.rs/agentai/badge.svg)](https://docs.rs/agentai)

<!-- cargo-rdme start -->

# AgentAI

AgentAI is a Rust library designed to simplify the creation of AI agents. It leverages
the [GenAI](https://crates.io/crates/genai) library to interface with a wide range of popular
Large Language Models (LLMs), making it versatile and powerful. Written in Rust, AgentAI
benefits from strong static typing and robust error handling, ensuring reliable
and maintainable code. Whether you're developing simple or complex AI agents, AgentAI provides
a streamlined and efficient development process.

## Warning

This library is under heavy development. The interface may change at any time without notice.

## Features

- **Connect to any major LLM provider**: Support for OpenAI, Anthropic, Gemini, Ollama, and other OpenAI-compatible APIs.
- **Choose the right model for the job**: Flexibly select the best-suited model for each step in your agent's workflow.
- **Build custom tools with ease**: A simple interface for creating and managing your own tools using the [`ToolBox`](https://docs.rs/agentai/latest/agentai/tool/trait.ToolBox.html).
- **MCP Server Support, local and remote**: Use tools from any Model Context Protocol server, whether you start it locally over **stdio** or connect to a running or hosted server over **Streamable HTTP** (with optional bearer token). Built on the official MCP Rust SDK.
- **Structured Output**: No need to parse raw text from model, just provide structure, and AI agent will provide response in defined format.

## What's New

#### Agent providers, remote MCP and configuration (version 0.2.0)

- **Creating agents**: `Agent::new_with_provider` binds an agent to any
  genai provider with an explicit API key, for example the native OpenRouter adapter, which needs no base URL:
  `Agent::new_with_provider(AdapterKind::OpenRouter, None, &api_key, system)?`. `Agent::new` and
  `Agent::new_with_url` now return `Result`, and the agent reports typed errors
  (`AgentError`). Tool errors are either recoverable (sent back to the model, which can
  try again) or fatal (they stop the run).
- **Remote MCP servers, not only stdio**: `McpToolBox` now connects to MCP
  servers over **Streamable HTTP**, so agents can use running and hosted servers, including ones that need a
  bearer token (`McpToolBox::new_http`, `McpToolBox::new_http_with_bearer`). Starting a local server over
  **stdio** (`McpToolBox::new`) works as before. MCP support is now built on `rmcp`, the official MCP Rust SDK.
- **Agent configuration**: `Agent::with_max_iterations` and
  `Agent::with_chat_options` set the loop limit and the model options
  (temperature, max tokens, tool choice, reasoning effort, ...).
- **Strict tools**: `#[tool(strict)]` marks a tool as strict, so providers that support it make the model's
  arguments match the parameter schema exactly.
- **Dependencies**: genai 0.7 (re-exported as `agentai::genai`; tool calls are no longer lost when a model
  also returns text), schemars 1, reqwest 0.13, syn 3, Rust edition 2024 and MSRV 1.88.

**Breaking changes**: `Agent::new`, `Agent::new_with_url` and `Agent::run` return `AgentError` instead of
`anyhow` errors, `ToolError` variants are renamed and split, `Tool` is built with `Tool::new`, and answer
and tool parameter types need `schemars` 1.x. Read the
[CHANGELOG](https://github.com/AdamStrojek/rust-agentai/blob/master/CHANGELOG.md) before upgrading: it lists
every breaking change and how to migrate.

#### `ToolBox` (version 0.1.5)

This release introduces the [`ToolBox`](https://docs.rs/agentai/latest/agentai/tool/trait.ToolBox.html), a new feature providing an easy-to-use interface for supplying tools to AI agents.

## Future Plans

We are continuously working to improve AgentAI. These are the next features we plan; the linked issues have the details:

- **Run details and history control** ([#46](https://github.com/AdamStrojek/rust-agentai/issues/46)): return token usage, the stop reason and reasoning from a run, and let you read and reset the conversation history.
- **Streaming output** ([#49](https://github.com/AdamStrojek/rust-agentai/issues/49)): stream the answer as it is generated, including runs with tool calls.
- **Agent Memory**: control what the agent remembers between runs, such as persistence, record limits and context management.
- **User input during a run**: let an agent ask the user for more information or confirmation before it continues.

## Installation

To add the AgentAI crate to your project, run the following command in your project's root directory:

```bash
cargo add agentai
```

This command adds the crate and its dependencies to your project.

## Feature Flags

<!-- FEATURE FLAGS -->

Available features for `agentai` crate.
To enable any of these features, you need to enter this command:

```bash
cargo add agentai -F mcp-client
```

Features list:

- `mcp-client` _(enabled by default)_ — Enables support for Agent Tools served by MCP servers (stdio and Streamable HTTP, via `rmcp`)
- `macros` _(enabled by default)_ — Enables support for macro [`#[toolbox]`](https://docs.rs/agentai/latest/agentai/tool/attr.toolbox.html)
- `tools-buildin` _(enabled by default)_ — Enables support for [buildin tools](https://docs.rs/agentai/latest/agentai/tool/buildin/index.html)
- `tools-web` _(enabled by default)_ — Enables support for [web tools](https://docs.rs/agentai/latest/agentai/tool/web/index.html)

## Usage

Here is a basic example of how to create an AI agent using AgentAI:

```rust
use agentai::Agent;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut agent = Agent::new("You are a useful assistant")?;
    let answer: String = agent.run("gpt-4o", "Why is the sky blue?", None).await?;
    println!("Answer: {}", answer);
    Ok(())
}
```

## Error handling

`Agent::run` returns an `AgentError` that you can match on. Provider failures that may
succeed on a second try report it with `AgentError::is_transient`:

```rust
use agentai::{Agent, AgentError};

match agent.run::<String>("gpt-4o", "Why is the sky blue?", None).await {
    Ok(answer) => println!("Answer: {answer}"),
    Err(err) if err.is_transient() => println!("Try again later: {err}"),
    Err(AgentError::InvalidAnswer { raw, .. }) => println!("Unexpected answer: {raw}"),
    Err(err) => println!("Failed: {err}"),
}
```

Tools report failures with [`ToolError`](https://docs.rs/agentai/latest/agentai/tool/enum.ToolError.html). Recoverable errors
(`NotFound`, `InvalidArguments`, `Failed`) are sent to the model as the tool result and the
run continues. Fatal errors (`Configuration`, `Fatal`) stop the run with
`AgentError::Tool`. [`ToolError::is_recoverable`](https://docs.rs/agentai/latest/agentai/tool/enum.ToolError.html#method.is_recoverable)
tells the two groups apart.

## Examples

For more examples, check out the [examples](https://docs.rs/agentai/latest/agentai/examples/) directory. To run an example, use the following command, replacing `<example_name>` with the name of the example file (without the `.rs` extension):

```bash
cargo run --example <example_name>
```

For instance, to run the `simple` example:

```bash
cargo run --example simple
```

<!-- cargo-rdme end -->

## Documentation

Full documentation is available on [docs.rs](https://docs.rs/agentai).

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the creators of the [GenAI library](https://crates.io/crates/genai) for providing a robust framework for interfacing with various LLMs.
