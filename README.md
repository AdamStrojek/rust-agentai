<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.png">
    <img src="assets/logo.png" alt="AgentAI logo" width="160">
  </picture>
</p>

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
- **MCP Server Support**: Leverage existing solutions based on the Model-Context-Protocol, eliminating the need to build agent tools from scratch.
- **Structured Output**: No need to parse raw text from model, just provide structure, and AI agent will provide response in defined format.

## What's New

#### genai 0.7 (version 0.2.0)

AgentAI now uses [genai](https://crates.io/crates/genai) 0.7, schemars 1.x and Rust edition 2024 (MSRV 1.88).
Models that return text together with tool calls now work: the tool calls are no longer dropped.
`Agent::new` and `Agent::new_with_url` now return `Result`, and a tool call without a toolbox
is answered with an error message to the model instead of a panic.
`McpToolBox` now uses the official MCP SDK (`rmcp`) and can also
connect to running servers over Streamable HTTP.
`Agent::new_with_provider` binds an agent to any genai provider,
for example the native OpenRouter adapter (`AdapterKind::OpenRouter`), with an explicit API key.
`Agent::with_max_iterations` and
`Agent::with_chat_options` configure the loop limit and the model
options (temperature, max tokens, tool choice, reasoning effort, ...). The `genai` crate is
re-exported as `agentai::genai`.
`#[tool(strict)]` marks a tool definition as strict, so supporting providers make the model's
arguments match the parameter schema exactly.

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
