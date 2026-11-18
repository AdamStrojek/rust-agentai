# Migration guide

How to upgrade code that uses `agentai` from one release to the next. Each section covers one jump
between consecutive releases, with the date of the newer release; the newest is at the top. If you
skip releases, apply the sections one after another, starting with the oldest.

The [CHANGELOG](CHANGELOG.md) lists every change; this guide covers only the ones that need changes
in your code.

- [0.7.0 → 0.7.1 (YYYY-MM-DD)](#070--071-yyyy-mm-dd)
- [0.6.0 → 0.7.0 (2026-11-18)](#060--070-2026-11-18)
- [0.4.0 → 0.5.0 (2026-11-04)](#040--050-2026-11-04)
- [0.3.1 → 0.4.0 (2026-10-26)](#031--040-2026-10-26)
- [0.2.0 → 0.3.0 (2026-10-22)](#020--030-2026-10-22)
- [0.1.5 → 0.2.0 (2026-10-20)](#015--020-2026-10-20)

## 0.7.0 → 0.7.1 (YYYY-MM-DD)

These changes are in `agentai-audio`.

### Voices are IDs only

`Voice::description(..)` and `Voice::auto(..)` are gone, and so are
`SpeechGenerator::supports_voice_description()` and the `descriptions_supported` field in the
answer of the `list_voices` tool. A voice is now always an ID, `Voice::id(..)`.

To get a voice from a description, design it once, keep a preview and use its ID. Fish Audio,
ElevenLabs and oMLX can design voices (`supports_voice_design()`).

```rust,ignore
// 0.7.0
let request = SpeechRequest::new("Hello")
    .with_voice(Voice::description("A warm, calm woman with a low voice"));

// 0.7.1
let description = "A warm, calm woman with a low voice";
let previews = generator.design_voice(description, Some("Hello"), 2).await?;
let id = generator.create_voice("narrator", &previews[0], Some(description)).await?;
let request = SpeechRequest::new("Hello").with_voice(Voice::id(id));
```

OpenAI cannot design voices: put the style in `instructions` instead, with
`SpeechRequest::with_instructions` (or the `instructions` argument of `generate_speech`).

`Voice::auto(x)` was a plain ID with a fallback to a description, and is now `Voice::id(x)`:

```rust,ignore
// 0.7.0
let voice = Voice::auto("nova");

// 0.7.1
let voice = Voice::id("nova");
```

With `SpeechToolBox`, a `voice` the model gives is now always sent as an ID. With `with_voices`, a
voice that is not in the list fails with a tool error the model can read, as before for providers
without descriptions.

### `create_voice` takes a `VoiceSource`

The samples and their transcript are one `VoiceSource`. A preview from `design_voice` is another
source, so `create_voice` also keeps a designed voice.

```rust,ignore
// 0.7.0
let id = generator
    .create_voice("me", &[sample], Some("What is said in the sample"), None)
    .await?;

// 0.7.1
let id = generator
    .create_voice(
        "me",
        VoiceSource::samples(vec![sample], Some("What is said in the sample")),
        None,
    )
    .await?;
```

Code that implements `SpeechProvider::create_voice` gets the same `VoiceSource` instead of
`samples` and `transcript`.

## 0.6.0 → 0.7.0 (2026-11-18)

### `ToolBox::call_tool` returns `ToolReturn`

Hand-written `ToolBox` implementations now return `ToolResult<ToolReturn>`. Wrap an immediate
`ToolOutput` in `ToolReturn` (or use `.map(Into::into)`). Methods using `#[toolbox]` that return
`ToolResult` or `ToolResult<ToolOutput>` need no change.

```rust
// 0.6.0
async fn call_tool(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
    Ok(ToolOutput::from_text("done"))
}

// 0.7.0
async fn call_tool(&self, call: &ToolCall) -> ToolResult<ToolReturn> {
    Ok(ToolOutput::from_text("done").into())
}
```

Code that calls `call_tool` directly must handle `ToolReturn::Done` and may also receive
`ToolReturn::Deferred`. Because `ToolReturn` is `#[non_exhaustive]`, include a wildcard arm:

```rust,ignore
match toolbox.call_tool(&call).await? {
    ToolReturn::Done(output) => use_output(output),
    ToolReturn::Deferred { message, task } => handle_deferred(message, task),
    _ => handle_future_tool_return(),
}
```

### Empty prompts are rejected

An empty or whitespace-only prompt now returns `AgentError::EmptyPrompt` unless the run has a
finished or cancelled deferred task to notify the model. The empty prompt is also omitted from
`AgentResult::new_messages`. Send non-empty text when you want a normal model response. For a
notification-only run, pass deferred task handles:

```rust,ignore
let result = agent
    .run(Prompt::new())
    .with_history(&history)
    .with_deferred_tasks(handles)
    .await?;
```

### Handle deferred tasks

`ToolReturn::Deferred` is new in 0.7.0. The agent assigns IDs in the
`<tool-call-id>-deferred-<sequence>` format, then appends the ID to the immediate tool response.
Use that ID—not `ToolCall::id`—with deferred-task management tools. Streaming consumers should
start tasks received in `AgentEvent::DeferredTask`; `Agent::run`, `AgentStream::finish` and the
listener helpers return unstarted tasks in `AgentResult::unstarted_deferred_tasks`.

The `tokio` feature now enables Tokio macros, runtime, synchronization, and time support. No code
changes are needed unless you depend on the exact dependency feature set.

`AgentResult` and `RunOptions` remain `#[non_exhaustive]`; their new deferred-task fields do not
require changes to code that uses the existing constructors and builders. New `AgentEvent`,
`RunEvent` and `AgentError` variants likewise do not require changes to exhaustive matches that
already include a wildcard arm.

### `agentai-image`, `agentai-audio`: one rule for file names

The name of a generated image or audio file, and of a file uploaded to a provider, keeps its
extension only when that is a known image, audio or video extension. Any other name gets the
extension of its type. Before, `agentai-image` kept every name that had a dot. A type without a
usable name, such as `application/octet-stream`, now gives `bin`.

```rust
// 0.6.0: saved as `my.picture`
// 0.7.0: saved as `my.picture.png`
let response = generator
    .generate_with(ImageRequest::new("A cat").with_name("my.picture"))
    .await?;

// To choose the name yourself, end it with a known extension
let request = ImageRequest::new("A cat").with_name("my-picture.png");
```

### `agentai-audio`: `SpeechRequest::voice` is now `voices`

A request can carry several voices, in speaker order, for dialogue models such as Fish Audio
`drama-3`, so the field `voice: Option<Voice>` is now `voices: Vec<Voice>`. `with_voice` adds a voice
to the list, so code that calls it once needs no change. Calling it again no longer replaces the
voice: it adds a second one, which providers with one voice (OpenAI, ElevenLabs, oMLX) reject with
`AudioError::Unsupported`. Use `with_voices(vec![..])` to replace the list. Code that reads or sets
the field directly:

```rust
// 0.6.0
let mut request = SpeechRequest::new("Hello");
request.voice = Some(Voice::id("alloy"));
if let Some(voice) = &request.voice { /* ... */ }
```

```rust
// 0.7.0
let mut request = SpeechRequest::new("Hello");
request.voices = vec![Voice::id("alloy")];
if let Some(voice) = request.voices.first() { /* ... */ }
```

A custom `SpeechProvider` that speaks with one voice should return `AudioError::Unsupported` when
`voices` has more than one entry, as the built-in OpenAI, ElevenLabs and oMLX providers do.

## 0.4.0 → 0.5.0 (2026-11-04)

### `call_tool` takes the `ToolCall` and returns `ToolOutput`

Tool results can carry images, so `ToolBox::call_tool` returns `ToolResult<ToolOutput>` instead of
`Result<String, ToolError>`. Text converts into `ToolOutput`. It also takes the whole call the model
made, a `&ToolCall` with `name`, `arguments` and `id` (the id the `ToolResponse` and the agent
events carry), instead of the name and the arguments.

Methods in a `#[toolbox]` that return `ToolResult` (text) need no change. A hand-written `ToolBox`:

```rust
// 0.4.0
async fn call_tool(&self, tool_name: String, arguments: Value) -> ToolResult {
    Ok("echoed".to_string())
}
```

```rust
// 0.5.0
use agentai::ToolCall;
use agentai::tool::ToolOutput;

async fn call_tool(&self, call: &ToolCall) -> ToolResult<ToolOutput> {
    // call.name, call.arguments (a serde_json::Value), call.id
    Ok("echoed".into())
}
```

Code that calls `call_tool` itself builds the call with `ToolCall::new(id, name, arguments)` and
reads the text with `.text()` (`Option<String>`).

### Tool results in the history and in events

`ToolResponse::content` is a `Vec<Part>` (text and media) instead of a `String`, and the `content` of
`AgentEvent::ToolResult` and `RunEvent::ToolResult` is a `ToolOutput` (`&ToolOutput`). Both print
with `{}` as before; read the text with `.text()`.

```rust
// 0.4.0
AgentEvent::ToolResult { content, .. } => log(&content),
```

```rust
// 0.5.0
AgentEvent::ToolResult { content, .. } => log(&content.text().unwrap_or_default()),
```

Histories stored with 0.4.0 still load: a tool result saved as a string becomes one text part.

### Choose a TLS backend

TLS is now chosen with features of `agentai` (and `agentai-image`): `rustls-ring` (default),
`rustls-aws-lc` (the backend of 0.4.0) or `native-tls`, plus `system-proxy` (default). If you set
`default-features = false`, add a backend, otherwise HTTPS requests fail:

```toml
# 0.4.0
agentai = { version = "0.4", default-features = false, features = ["macros"] }
# 0.5.0
agentai = { version = "0.5", default-features = false, features = ["macros", "rustls-ring", "system-proxy"] }
```

With `rustls-ring`, building a `reqwest` client panics until a rustls crypto provider is installed.
`agentai` installs ring before it builds its own clients; if you build a `genai::Client` for
`Agent::from_client` or your own `reqwest::Client` before that, install it first:

```rust
// 0.5.0
agentai::tls::install_crypto_provider();
let client = agentai::genai::Client::builder()/* ... */.build()?;
let agent = Agent::from_client(client, "gpt-4.1-mini", "You are a useful assistant");
```

## 0.3.1 → 0.4.0 (2026-10-26)

### `Agent::new` takes the model

The model is passed to `Agent::new` together with the system prompt, so an agent always has one.

```rust
// 0.3.1
let agent = Agent::new(SYSTEM)?.with_model(&model);
```

```rust
// 0.4.0
let agent = Agent::new(&model, SYSTEM)?;
```

The system prompt may be empty (`""`); then none is sent. To build an agent without a model (and
set it later or on each run), use `Agent::default()`:

```rust
// 0.4.0
let agent = Agent::default().with_model(&model).with_system(SYSTEM);
```

### `Agent::new_with_provider` and `Provider` are removed

For a provider genai knows, put its namespace in front of the model name and set the provider's
API key environment variable, for example `OPEN_ROUTER_API_KEY` for OpenRouter.

```rust
// 0.3.1
let agent = Agent::new_with_provider(Provider::OpenRouter, None, &api_key, SYSTEM)?.with_model(&model);
```

```rust
// 0.4.0, with OPEN_ROUTER_API_KEY set
let agent = Agent::new(&format!("open_router::{model}"), SYSTEM)?;
```

For anything else (a custom endpoint, an explicit key), build the client yourself as shown below.

### `Agent::new_with_url` is removed

Build a genai client for the endpoint with `agentai::genai` and pass it to `Agent::from_client`.

```rust
// 0.3.1
let agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?.with_model(&model);
```

```rust
// 0.4.0
use agentai::genai::Client;
use agentai::genai::adapter::AdapterKind;
use agentai::genai::resolver::{AuthData, Endpoint};

let client = Client::builder()
    .with_adapter_kind(AdapterKind::OpenAI)
    .append_provider_config(
        AdapterKind::OpenAI,
        (Endpoint::from_owned(base_url), AuthData::from_single(api_key)),
    )
    .build()?;
let agent = Agent::from_client(client, &model, SYSTEM);
```

## 0.2.0 → 0.3.0 (2026-10-22)

### Set the model and the toolbox on the agent

`run` takes only the prompt. The model and the toolbox belong to the agent, and `run` takes `&self`,
so the agent no longer needs `mut` and can be cloned and shared between tasks. The toolbox is passed
by value; to share one between agents, wrap it in an `Arc` and use `with_toolbox_arc`.

```rust
// 0.2.0
let mut agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?;
let answer: String = agent.run(&model, question, Some(&tools)).await?;
```

```rust
// 0.3.0
let agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?
    .with_model(&model)
    .with_toolbox(tools);
let result: AgentResult<String> = agent.run(question).await?;
let answer = result.answer;
```

A run without a model fails with `AgentError::MissingModel`. To use another model or toolbox for one
run, set it on the request:

```rust
// 0.2.0
let answer: String = agent.run("gpt-4.1-mini", question, None).await?;
```

```rust
// 0.3.0
let result: AgentResult<String> = agent.run(question).with_model("gpt-4.1-mini").await?;
```

### Read the answer from `AgentResult`

A run returns `AgentResult<D>` instead of `D`. The answer is in `answer`; next to it are
`new_messages`, `usage`, `stop_reason`, `reasoning_content` and `iterations`.

```rust
// 0.2.0
let answer: Weather = agent.run(&model, question, None).await?;
println!("{}", answer.temperature);
```

```rust
// 0.3.0
let result: AgentResult<Weather> = agent.run(question).await?;
println!("{}", result.answer.temperature);
println!("{} tokens", result.usage.total_tokens);
```

### Keep the conversation history yourself

The agent no longer remembers earlier runs. In 0.2.0, a second `run` on the same agent continued the
conversation; in 0.3.0 every run starts from the history you pass. Keep a `Vec<Message>`, pass it with
`with_history` and add the run's `new_messages` to it:

```rust
// 0.2.0
let mut agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?;
let first: String = agent.run(&model, "Why is the sky blue?", None).await?;
let second: String = agent.run(&model, "And at sunset?", None).await?; // continues
```

```rust
// 0.3.0
let agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?.with_model(&model);
let mut history: Vec<Message> = Vec::new();

let result: AgentResult<String> = agent.run("Why is the sky blue?").with_history(&history).await?;
history.extend(result.new_messages);

let result: AgentResult<String> = agent.run("And at sunset?").with_history(&history).await?;
history.extend(result.new_messages);
```

The system prompt is not part of the history: the agent sends it with every request. A failed run
does not change your history, so you can retry with the same one. `Message` implements `Serialize`
and `Deserialize`, so the history can be stored and loaded.

To start a new conversation, use an empty history. There is no need to create a new agent.

### Handle the new errors

An answer cut off by the token limit fails with `AgentError::MaxTokens { raw }` instead of
`AgentError::InvalidAnswer`. Tool calls in a cut-off answer are not run.

```rust
// 0.2.0
match agent.run::<Weather>(&model, question, None).await {
    Err(AgentError::InvalidAnswer { raw, .. }) => { /* also a truncated answer */ }
    // ...
}
```

```rust
// 0.3.0
match agent.run::<Weather>(question).await {
    Err(AgentError::MaxTokens { raw }) => { /* raise max_tokens or ask for less */ }
    Err(AgentError::InvalidAnswer { raw, .. }) => { /* the answer does not match the type */ }
    // ...
}
```

`AgentError::Llm` and `AgentError::ClientInit` hold an `LlmError`. `AgentError::is_transient()` works
as before; `LlmError` also has `status()` with the HTTP status:

```rust
// 0.3.0
if let Err(AgentError::Llm(err)) = &outcome {
    eprintln!("provider answered {:?}, retry: {}", err.status(), err.is_transient());
}
```

### `Agent::new_with_provider` takes a `Provider`

```rust
// 0.2.0
use agentai::{AdapterKind, Agent};
let agent = Agent::new_with_provider(AdapterKind::OpenRouter, None, &api_key, SYSTEM)?;
```

```rust
// 0.3.0
use agentai::{Agent, Provider};
let agent = Agent::new_with_provider(Provider::OpenRouter, None, &api_key, SYSTEM)?.with_model(&model);
```

### `Agent::new_with_client` is removed

An agent can no longer be built from your own LLM client. Use `Agent::new` (provider picked from the
model name, API key from the provider's environment variable), `Agent::new_with_url` (any
OpenAI-compatible endpoint) or `Agent::new_with_provider` (one provider with an explicit API key and
optional endpoint).

### Import `ToolChoice` and `ReasoningEffort` from `agentai`

```rust
// 0.2.0
use agentai::genai::chat::{ChatOptions, ReasoningEffort, ToolChoice};
```

```rust
// 0.3.0
use agentai::{ChatOptions, ReasoningEffort, ToolChoice};
```

### New: streaming

Not a breaking change, but it replaces workarounds: `run_stream` takes the same configuration as
`run` and returns the answer as it is written, with the tool calls the agent makes. Reading the
stream needs `StreamExt` from the `futures-util` (or `futures`) crate.

```rust
// 0.3.0
use futures_util::StreamExt;

let mut stream = agent.run_stream::<String>(question).with_history(&history).await?;
while let Some(event) = stream.next().await {
    match event? {
        AgentEvent::TextDelta(text) => print!("{text}"),
        AgentEvent::Finished(result) => history.extend(result.new_messages),
        _ => {}
    }
}
```

## 0.1.5 → 0.2.0 (2026-10-20)

### Toolchain and dependencies

- The minimum Rust version is 1.88, and the crates use edition 2024.
- Answer types and tool parameter types derive `JsonSchema` from `schemars` 1.x (was 0.9). Update
  `schemars` in your `Cargo.toml`:

  ```toml
  # 0.1.5
  schemars = { version = "0.9", features = ["derive"] }
  # 0.2.0
  schemars = { version = "1", features = ["derive"] }
  ```

- `reqwest` 0.13 uses rustls by default (was native-tls).

### Constructors return `Result`

```rust
// 0.1.5
let mut agent = Agent::new_with_url(&base_url, &api_key, SYSTEM);
```

```rust
// 0.2.0
let mut agent = Agent::new_with_url(&base_url, &api_key, SYSTEM)?;
```

The same applies to `Agent::new`.

### `run` returns `AgentError` instead of `anyhow::Error`

`?` inside a function that returns `anyhow::Result` keeps working. To react to errors, match on
`AgentError`:

```rust
// 0.1.5
let answer: String = agent.run(&model, question, None).await?; // anyhow::Error
```

```rust
// 0.2.0
match agent.run::<String>(&model, question, None).await {
    Ok(answer) => println!("{answer}"),
    Err(err) if err.is_transient() => { /* retry later */ }
    Err(AgentError::InvalidAnswer { raw, .. }) => eprintln!("unexpected answer: {raw}"),
    Err(err) => return Err(err.into()),
}
```

With `new_with_url`, a model name with another provider's namespace (for example `anthropic::...`)
fails with an adapter mismatch error. Plain names such as `vendor/model:free` work as before.

### Return a `ToolError` from tools

There is no `From<anyhow::Error>` for `ToolError` any more. Map errors in tool methods to a
`ToolError` variant. Recoverable errors (`NotFound`, `InvalidArguments`, `Failed`) are sent to the
model, which can try again; fatal errors (`Configuration`, `Fatal`) stop the run with
`AgentError::Tool`.

```rust
// 0.1.5
#[tool]
async fn web_fetch(&self, url: String) -> ToolResult {
    Ok(reqwest::get(url)
        .await
        .map_err(anyhow::Error::new)?
        .text()
        .await
        .map_err(anyhow::Error::new)?)
}
```

```rust
// 0.2.0
#[tool]
async fn web_fetch(&self, url: String) -> ToolResult {
    reqwest::get(&url)
        .await
        .map_err(|e| ToolError::Failed(format!("Request to {url} failed: {e}")))?
        .text()
        .await
        .map_err(|e| ToolError::Failed(format!("Reading {url} failed: {e}")))
}
```

Renamed and removed `ToolError` variants:

| 0.1.5 | 0.2.0 |
|---|---|
| `NoToolFound(name)` | `NotFound(name)` |
| `LLMError(message)` | `InvalidArguments(message)` or `Failed(message)` |
| `ToolsDefinitionNotReady` | `Configuration(message)` |
| `ExecutionError` | `Failed(message)` or `ToolError::fatal(err)` |
| `Other(anyhow::Error)` | `ToolError::fatal(err)` |

`ToolError` is `#[non_exhaustive]`, so a `match` on it needs a `_` arm.

`#[toolbox]` no longer replaces the errors your tools return with `ExecutionError`: they reach the
model as they are, and arguments that do not deserialize become `InvalidArguments`.
`McpToolBox::call_tool` returns `NotFound` for tools the server does not have and `Failed` for errors
the server reports; in 0.1.5 it returned `Ok` with the error text.

### Build `Tool` with `Tool::new`

A tool definition can no longer be built with a struct literal:

```rust
// 0.1.5
Tool {
    name: "echo".to_string(),
    description: Some("Returns the given text".to_string()),
    schema: Some(schema),
}
```

```rust
// 0.2.0
Tool::new("echo")
    .with_description("Returns the given text")
    .with_schema(schema)
```

Compare tool names with `tool.name.as_str()`.
