# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.4](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.7.3...agentai-v0.7.4) - 2026-11-19

### Breaking changes

See [MIGRATION.md](MIGRATION.md#073--074-2026-11-19) for migration steps from 0.7.3.

- *(audio)* `VoiceSource::Samples` has a `language` field and is `#[non_exhaustive]`: build it with `VoiceSource::samples(..)` and `with_language`, and match it with `..` (custom `create_voice` implementations).

### New features

- *(audio)* design a voice in a language with `design_voice(..).with_language("pl")`, sent to Fish Audio and oMLX (ElevenLabs has no such setting); `VoicePreview::language` is the language of a preview, and keeping it with `create_voice` stores it with the voice ([#103](https://github.com/AdamStrojek/rust-agentai/pull/103)) (by @AdamStrojek)
- *(audio)* `StoredVoice::language` and `VoiceInfo::language`, filled from the store, Fish Audio model languages and ElevenLabs labels (by @AdamStrojek)

### Miscellaneous

- release notes for 0.7.3 ([#102](https://github.com/AdamStrojek/rust-agentai/pull/102)) (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.7.3](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.7.2...agentai-v0.7.3) - 2026-11-19

### New features
- *(audio)* set the speaking speed of a dialogue with `DialogueRequest::with_speed`: Fish Audio sends it as `prosody`, ElevenLabs dialogues return `Unsupported` ([#100](https://github.com/AdamStrojek/rust-agentai/pull/100)) (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.7.2](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.7.1...agentai-v0.7.2) - 2026-11-19

### Breaking changes

All in `agentai-audio`, released as a patch while the audio interface is reworked. See [MIGRATION.md](MIGRATION.md#071--072-2026-11-19) for migration steps from 0.7.1.

- *(audio)* Build a provider with `speech_provider(provider, speech_model)` (or `SpeechProviderKind::X.builder(..)`) and pass it to `SpeechGenerator::new`. `SpeechGenerator::new(model)`, `provider::from_model` and `provider::list_models(name)` are removed.
- *(audio)* Calls are builders you `await`: `generate(text).with_voice(id)...`, `design_voice(description).with_text(..).with_count(n)`. `generate_with` and the public fields of `SpeechRequest` are gone, and so are the generator defaults (`with_default_voice`, `with_format`, `with_speed`, `with_language`). Without `with_voice` no voice is sent.
- *(audio)* Dialogues use `generate_dialogue` and the provider's dialogue model (`with_dialogue_model`) instead of `SpeechRequest::with_voices` and speaker tags in the text.
- *(audio)* Voice design needs a design model (`with_design_model`): Fish Audio no longer defaults to `voice-design-1`, and oMLX no longer designs with its speech model.
- *(audio)* `Voice` is replaced by `VoiceId` (a `String`). `SpeechProvider::model()` and `AudioModel::model` are removed, and custom providers read requests through getters.
- *(audio)* `VoiceInfo::cloned` is replaced by `origin` (`Preset`, `Account`, `Store`), and `VoiceInfo::new` takes it. The `list_voices` tool reports `"origin"` instead of `"cloned"`.

### New features

- *(audio)* speak dialogues as a list of turns with `generate_dialogue([DialogueTurn::new(voice, text), ..])`, with Fish Audio (S2 family, `drama-3`) and ElevenLabs (`/v1/text-to-dialogue`); you no longer write speaker tags ([#98](https://github.com/AdamStrojek/rust-agentai/pull/98)) (by @AdamStrojek)
- *(audio)* a separate model for speech, dialogues and voice design; an engine without its model returns `MissingModel` without a request, and `with_model` overrides the model for one call (by @AdamStrojek)
- *(audio)* `speech_provider("fishaudio", "s2-pro")` builds any provider by name, and `SpeechProviderKind::known_models()` lists the known models with what each can do (speech, dialogue, voice design), without a request (by @AdamStrojek)
- *(audio)* `VoiceInfo::origin` tells account voices from `VoiceStore` voices, and `VoiceInfo::preview` is a recording to play (ElevenLabs `preview_url`, a Fish Audio model sample, a store voice's sample) (by @AdamStrojek)

### Fixed

- *(audio)* a Fish Audio dialogue whose voices came from a `VoiceStore` sent both the store IDs and the samples, so the server could speak turns in the wrong voice; a request now sends samples when all its voices are in the store, voice model IDs when none are, and fails with `InvalidRequest` for a mix (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.7.1](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.7.0...agentai-v0.7.1) - 2026-11-18

### Breaking changes

See [MIGRATION.md](MIGRATION.md#070--071-2026-11-18) for migration steps from 0.7.0.

- *(audio)* `Voice::description` and `Voice::auto` are removed: a voice is always an ID (`Voice::id`). Design a voice from a description with `design_voice` and keep it with `create_voice`. `supports_voice_description()` and `descriptions_supported` are gone too.
- *(audio)* `create_voice` takes a `VoiceSource` (`VoiceSource::samples(samples, transcript)` or a designed preview) instead of `samples` and `transcript`.
- *(audio)* `SpeechToolBox`: `generate_speech` takes only voice IDs; with `with_voices`, a voice that is not in the list fails as a recoverable tool error instead of being used as a description.

### New features

- *(audio)* design voices from a description with `SpeechGenerator::design_voice`, which returns `VoicePreview`s to listen to; `create_voice` keeps one as a voice. Works with Fish Audio, ElevenLabs and oMLX; `supports_voice_design()` tells whether the provider can (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.7.0](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.6.0...agentai-v0.7.0) - 2026-11-18

### Breaking changes

See [MIGRATION.md](MIGRATION.md#060--070-2026-11-18) for migration steps from 0.6.0.

- *(tool)* `ToolBox::call_tool` returns `ToolResult<ToolReturn>` instead of `ToolResult<ToolOutput>`. Hand-written implementations and direct callers must handle the non-exhaustive `ToolReturn`; `#[toolbox]` methods need no change.
- *(audio)* `SpeechRequest::voice: Option<Voice>` is now `voices: Vec<Voice>`, and calling `with_voice` twice adds a second voice instead of replacing the first.
- *(agent)* An empty or whitespace-only prompt returns `AgentError::EmptyPrompt` when there are no finished deferred tasks to report, and is no longer added to `new_messages`.
- *(image, audio)* A generated or uploaded file keeps the extension of its name only when it is a known image, audio or video extension; any other name gets the extension of its type (`my.picture` is saved as `my.picture.png`), and a type without a usable extension gives `bin`.

### New features

- *(agent)* run tools in the background: a tool returns `ToolReturn::deferred(message, future)`, the model gets the message and a task ID at once, and the agent tells the model when the task finishes, in the same run or a later one through `with_deferred_tasks` (by @sdi2200246)
- *(video)* add the agentai-video crate: submit video jobs, poll their status and download the video with comfy-serve or OpenRouter, from text, first and last frames or reference images, videos and audio; `list_models` tells what a model accepts, and a serializable `VideoJob` can be followed after a restart (by @AdamStrojek)
- *(tool)* add `TokioDeferredTaskToolBox` (feature `tokio`): spawns deferred tasks and gives the model `list_deferred_tasks`, `get_deferred_task` and `wait_deferred_task` (by @sdi2200246)
- *(agent)* trim what is sent to the model with history processors: `Agent::with_history_processor` takes a `HistoryProcessor` or a closure, and the ready-made `HistoryPruning` drops images and reasoning from finished turns; your history stays complete (by @AdamStrojek)
- *(agent)* send provider-specific headers and body fields with `ChatOptions::with_extra_header` and `with_extra_body` (by @AdamStrojek)
- *(agent)* cap the dollar cost of runs with `Agent::with_budget` (feature `budget`, built on `llm-budget`); `AgentResult::cost_usd` reports what a run cost (by @Mattbusel)
- *(audio)* add the Fish Audio provider (`fishaudio::<model>`): dialogues with several voices in one request, voice cloning, voices from a `VoiceStore`, and word timestamps with `SpeechRequest::with_timestamps` (by @AdamStrojek)
- *(core)* add `agentai_core::mime` (MIME types to file extensions and back, re-exported by agentai-image, agentai-audio and agentai-video) and `Media::extension`, `Media::file_name` and `Media::to_url` (by @AdamStrojek)

### Changed

- *(agent)* finished and cancelled deferred tasks found in one loop check are reported to the model in a single user message (by @sdi2200246)
- *(feature)* the `tokio` feature now also enables Tokio's runtime, synchronization, time and macro support (by @sdi2200246)

### Miscellaneous

- *(core)* HTTP requests support `DELETE` and JSON parts in multipart forms; the media crates share the MIME table, base64 helpers and a `MockHttp` test client (feature `test-support`) (by @AdamStrojek)
- *(ci)* run LLM examples on pull requests only with the `live-tests` label, and review pull requests from forks (by @AdamStrojek)

### Contributors

* @AdamStrojek
* @Mattbusel
* @sdi2200246

## [0.6.0](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.5.0...agentai-v0.6.0) - 2026-11-09

### New features
- *(audio)* add the agentai-audio crate: text-to-speech with OpenAI, ElevenLabs and oMLX (by @AdamStrojek)
- *(audio)* clone voices with create_voice on ElevenLabs and oMLX (by @AdamStrojek)
- *(audio)* let agents speak with SpeechToolBox: generate_speech and list_voices, with an allowlist of voices (by @AdamStrojek)
- *(core)* add HttpRequest::with_bearer_token, HttpResponse::error_for_status, Media::bytes and Media::byte_len (by @AdamStrojek)

### Changed
- *(image)* share the HTTP and media helpers of agentai-core (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.5.0](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.4.0...agentai-v0.5.0) - 2026-11-04

### New features
- *(tool)* parse arrays and objects sent as JSON strings in #[toolbox] tools (by @AdamStrojek)
- *(image)* list image models and generate images with OpenRouter (by @AdamStrojek)
- *(tls)* choose the TLS backend with cargo features, ring by default (by @AdamStrojek)
- *(image)* generate and edit images with the agentai-image toolbox (by @AdamStrojek)
- *(core)* add the HTTP client abstraction in agentai-core (by @AdamStrojek)
- *(image)* let the model name the images it makes (by @AdamStrojek)
- *(image)* re-export the media storage as agentai_image::storage (by @AdamStrojek)

### Changed
- *(tool)* send tool images in a user message for every provider (by @AdamStrojek)
- *(tool)* pass the ToolCall with its id to ToolBox::call_tool (by @AdamStrojek)
- *(image)* expose ImageGenerator and split the image toolboxes (by @AdamStrojek)
- *(core)* move the message types to agentai-core and add MediaStorage (by @AdamStrojek)
- *(tool)* [**breaking**] let tools return images with ToolOutput (by @AdamStrojek)
- *(image)* build the image toolboxes with #[toolbox], name required (by @AdamStrojek)

### Fixes
- *(tool)* show tool images to OpenAI-compatible models (by @AdamStrojek)

### Miscellaneous
- *(tool)* say that tools convert their data to text themselves (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.4.0](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.3.2...agentai-v0.4.0) - 2026-10-26

### Changed
- *(agent)* [**breaking**] require the model in Agent::new and build custom clients with from_client ([#79](https://github.com/AdamStrojek/rust-agentai/pull/79)) (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.3.2](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.3.1...agentai-v0.3.2) - 2026-10-26

### New features
- *(agent)* send images in the prompt to vision models ([#77](https://github.com/AdamStrojek/rust-agentai/pull/77)) (by @AdamStrojek)

### Contributors

* @AdamStrojek

## [0.3.1](https://github.com/AdamStrojek/rust-agentai/compare/agentai-v0.3.0...agentai-v0.3.1) - 2026-10-22

### New features
- *(agent)* print a streamed run and follow it with event listeners ([#71](https://github.com/AdamStrojek/rust-agentai/pull/71)) (by @AdamStrojek)

### Contributors

* @AdamStrojek

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
