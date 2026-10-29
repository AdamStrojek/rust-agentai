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
- **Streaming**: Show the answer as the model writes it, together with the tool calls the agent makes. The stream is `Send + 'static`, ready for spawned tasks and server-sent events.
- **Images for vision models**: Send text and images together in a `Prompt`, by URL or from bytes. Plain text prompts work as before.
- **Image generation and editing**: The [`agentai-image`](https://docs.rs/agentai-image) crate gives the agent tools to draw and edit images (OpenAI and compatible APIs). Tools can return images to the model with `ToolOutput`.
- **You own the conversation**: An agent is shareable configuration. You keep the history, decide what to store, trim or persist, and get token usage and the stop reason from every run.

## What's New

#### 0.5.0 - Image generation, images in tool results

- **Image generation and editing**: the new [`agentai-image`](https://docs.rs/agentai-image) crate has
  `ImageGenerator` to make and edit images from your code: `ImageGenerator::new("openai::gpt-image-1")?`
  takes the provider from the model name and the key from `OPENAI_API_KEY`; OpenAI-compatible servers work
  with `OPENAI_BASE_URL` or a base URL set in code.
- **Agents that draw**: `GenerateImageToolBox` gives the agent `generate_image` and `edit_image`. The model
  gets the image itself, or only its metadata as JSON to save tokens. `ReadMediaToolBox` lets the model read
  any stored file by its ID.
- **Media storage**: `MediaRead` and `MediaSave` (in `agentai-core`) store images and other files under an
  ID; implement them for your blob store or directory, or use `MemoryStorage`.
- **Tools can return images**: a tool returns a `ToolOutput` with text and
  `Media` parts, for example a chart or a screenshot. Providers that accept images in tool
  results (Anthropic, Gemini, OpenAI Responses, Bedrock, Ollama) get them, others get the text. `#[toolbox]`
  methods return `ToolResult<ToolOutput>` for that; methods that return text stay as they are.
- Internal: the new `agentai-core` crate holds the shared types (`Message`, `Prompt`, `Media`, still
  available from `agentai` under the same paths) and the HTTP client abstraction used by `agentai-image`, so it
  can later run on other HTTP stacks, such as WebAssembly.

**Breaking changes**: `ToolBox::call_tool` returns `ToolResult<ToolOutput>`, and `ToolResponse::content` and
the `content` of tool result events are no longer a `String`. Read the
[CHANGELOG](https://github.com/AdamStrojek/rust-agentai/blob/master/CHANGELOG.md) and the
[migration guide](https://github.com/AdamStrojek/rust-agentai/blob/master/MIGRATION.md#040--050-YYYY-MM-DD)
before upgrading.

#### 0.3.0 - Streaming, conversation history in your hands, run details

- **Streaming**: `Agent::run_stream` takes the same configuration as
  `Agent::run`, and awaiting it returns a stream of `AgentEvent`s:
  text and reasoning as the model writes them, each tool call and its result, and finally the result of
  the run. Runs with tool calls and structured output stream too. The stream is `Send + 'static`, so you
  can spawn it or return it from a web handler.
- **You manage the history**: the agent no longer keeps the conversation. Pass your history with
  `.with_history(&history)` and add the run's `new_messages` to it when you
  want to continue: `history.extend(result.new_messages)`. Store, trim, persist or reset it however you
  like: `Message` has its own stable JSON format. A failed or cancelled run never changes
  your history, and the system prompt is not part of it: the agent sends it with every request.
- **One agent, many conversations**: the model and toolbox now belong to the agent
  (`Agent::with_model`, `Agent::with_toolbox`), and
  running takes `&self`, so one agent can be cloned cheaply and shared between tasks. Change the model,
  toolbox, iteration limit or options for a single run with the `with_*` methods of
  `AgentRequest` or a reusable `RunOptions`.
- **Run details**: awaiting a run returns a `AgentResult` with the answer, token usage summed
  over all model calls, the stop reason, the model's reasoning and the number of model calls. An answer cut
  off by the token limit is reported as `AgentError::MaxTokens` instead of a
  confusing parse error.

**Breaking changes**: `agent.run(model, prompt, toolbox)` is replaced by
`agent.run(prompt).with_history(&history).await`, which returns an `AgentResult` instead of the bare answer;
the model and toolbox are set with `Agent::with_model` and `Agent::with_toolbox` (or per run), and the agent
no longer remembers earlier runs. `Agent::new_with_provider` takes a `Provider`, and `Agent::new_with_client`
is removed. Follow the [migration guide](https://github.com/AdamStrojek/rust-agentai/blob/master/MIGRATION.md) to upgrade from 0.2.0: it shows each change with before and
after code.

#### 0.2.0 - Agent providers, remote MCP and configuration

- **Creating agents**: `Agent::new_with_provider` binds an agent to any
  genai provider with an explicit API key, for example the native OpenRouter adapter, which needs no base URL:
  `Agent::new_with_provider(AdapterKind::OpenRouter, None, &api_key, system)?`. `Agent::new` and
  `Agent::new_with_url` now return `Result`, and the agent reports typed errors
  (`AgentError`). Tool errors are either recoverable (sent back to the model, which can
  try again) or fatal (they stop the run).
- **Remote MCP servers, not only stdio**: [`McpToolBox`](https://docs.rs/agentai/latest/agentai/tool/mcp/struct.McpToolBox.html) now connects to MCP
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

#### 0.1.5 - `ToolBox`

This release introduces the [`ToolBox`](https://docs.rs/agentai/latest/agentai/tool/trait.ToolBox.html), a new feature providing an easy-to-use interface for supplying tools to AI agents.

## Future Plans

We are continuously working to improve AgentAI. These are the next features we plan; the linked issues have the details:

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

## Usage

Here is a basic example of how to create an AI agent using AgentAI:

```rust
use agentai::{Agent, Message, AgentResult};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let agent = Agent::new("gpt-4o", "You are a useful assistant")?;

    // The conversation history is yours
    let mut history: Vec<Message> = Vec::new();

    let result: AgentResult<String> = agent.run("Why is the sky blue?").with_history(&history).await?;
    println!("Answer: {}", result.answer);
    // Keep the question and the answer to continue the conversation
    history.extend(result.new_messages);
    Ok(())
}
```

Models that support images take a `Prompt` with text and images, by URL or from bytes:

```rust
use agentai::{Agent, AgentResult, Prompt};

let prompt = Prompt::text("Which of these logos is brighter?")
    .with_image_url("https://example.com/logo.png")
    .with_image_bytes("image/png", &png);
let result: AgentResult<String> = agent.run(prompt).await?;
```

To show the answer while it is written, use `run_stream` with the same configuration and
write the stream to the console, a file or any other `std::io::Write`:

```rust
use agentai::{Agent, AgentResult};

let result: AgentResult<String> = agent
    .run_stream("Why is the sky blue?")
    .await?
    .print_to(std::io::stdout())
    .await?;
```

For your own handling, read the [`AgentEvent`]s of the stream:

```rust
use agentai::{Agent, AgentEvent};
use futures_util::StreamExt;

let mut stream = agent.run_stream::<String>("Why is the sky blue?").await?;
while let Some(event) = stream.next().await {
    match event? {
        AgentEvent::TextDelta(text) => print!("{text}"),
        AgentEvent::Finished(result) => println!("\n{} tokens", result.usage.total_tokens),
        _ => {}
    }
}
```

## Error handling

A run returns an [`AgentError`] that you can match on. Provider failures that may
succeed on a second try report it with [`AgentError::is_transient`]. Your history is not
changed by a failed run, so you can simply try again:

```rust
use agentai::{Agent, AgentError};

match agent.run::<String>("Why is the sky blue?").await {
    Ok(result) => println!("Answer: {}", result.answer),
    Err(err) if err.is_transient() => println!("Try again later: {err}"),
    Err(AgentError::InvalidAnswer { raw, .. }) => println!("Unexpected answer: {raw}"),
    Err(err) => println!("Failed: {err}"),
}
```

Tools report failures with [`ToolError`](https://docs.rs/agentai/latest/agentai/tool/enum.ToolError.html). Recoverable errors
(`NotFound`, `InvalidArguments`, `Failed`) are sent to the model as the tool result and the
run continues. Fatal errors (`Configuration`, `Fatal`) stop the run with
[`AgentError::Tool`]. [`ToolError::is_recoverable`](https://docs.rs/agentai/latest/agentai/tool/enum.ToolError.html#method.is_recoverable)
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
