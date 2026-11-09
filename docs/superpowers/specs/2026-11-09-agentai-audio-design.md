# `agentai-audio`: text-to-speech and voice cloning

## Context

agentai can generate images (`agentai-image`) but cannot produce speech. This adds a new workspace crate, `crates/agentai-audio`, that synthesizes speech and clones voices, usable both directly from application code and as agent tools. Layout follows `agentai-image` so users meet the same provider / generator / toolbox shape.

Provider facts below come from primary docs and source read on 2026-11-09. Two affect the design:
- OpenAI shuts down every `/v1/audio/speech` model on 2027-01-06; the suggested successor is Realtime-only. The OpenAI wire format stays useful as the shape oMLX copies.
- A cloned voice is a server-side object on ElevenLabs (and OpenAI), but oMLX has no voice object and takes the sample on every request.

## Goals

- Generate speech from text through one provider abstraction: OpenAI, ElevenLabs, oMLX (test model `mlx-community/VoxCPM2-8bit`).
- Clone a voice with the same trait call on every provider that can: `provider.create_voice(CreateVoice::new("Adam", vec![sample]))` returns an ID, used through `Voice::id(id)`. On oMLX the provider must be built explicitly with a voice store first (see `VoiceStore`).
- List voices on the abstract provider and through an agent tool.
- Agent tool `generate_speech` with one optional `voice` parameter whose meaning depends on the provider: library voice, cloned voice ID, or a free-text description of timbre or gender.

## Non-goals (first version)

- Streaming playback: `generate` returns the finished audio. A later `generate_stream` trait method can add it without breaking the API. No `HttpClient` changes.
- Creating custom voices on OpenAI (limited-access program, consent recording, service ends soon). `create_voice` returns `Unsupported`; an existing `voice_...` ID can still be used.
- Deleting voices (`HttpClient` has only GET and POST).
- Speech recognition, music, sound effects, Realtime APIs. (Assumed out of scope from "TTS".)

## Anti-goals

- The LLM must not be able to create a clone. Cloning is application code only; no tool reaches `create_voice`. Reason: cloning needs the speaker's consent, and web content can steer an agent.
- No model parameters invented by the library: required values such as OpenAI's `voice` come from the caller and otherwise fail clearly.
- `cargo tree -i aws-lc-rs` stays empty; the crate stays runtime-neutral (no tokio, clock or randomness in the core path).

## Design

### Crate setup

`crates/agentai-audio`, modelled on `crates/agentai-image/Cargo.toml`: features `toolbox` (default; `dep:agentai`, `agentai/macros`, `dep:schemars`), `rustls-ring` (default), `rustls-aws-lc`, `native-tls`, `system-proxy` (default), forwarded to `agentai-core` and `agentai?`. Re-export `agentai_core::storage` as `agentai_audio::storage`; providers are not re-exported at the root.

### `provider/mod.rs`

```rust
#[async_trait]
pub trait SpeechProvider: Send + Sync + 'static {
    fn model(&self) -> Option<&str> { None }
    async fn generate(&self, request: SpeechRequest) -> Result<SpeechResponse, AudioError>;
    fn supports_voice_cloning(&self) -> bool { false }
    /// Returns the ID to use with `Voice::id`.
    async fn create_voice(&self, request: CreateVoice) -> Result<String, AudioError>; // default Unsupported
    fn supports_voice_description(&self) -> bool { false }
    async fn list_voices(&self) -> Result<Vec<VoiceInfo>, AudioError>; // default Unsupported
    async fn list_models(&self) -> Result<Vec<AudioModel>, AudioError>; // default Unsupported
}
```

All structs `#[non_exhaustive]`, `new` plus `with_*`.

- `Voice` (`#[non_exhaustive]` enum): `Id(String)` (`Voice::id`), `Description(String)` (`Voice::description`), `Auto(String)` (`Voice::auto`: the provider decides; what the tool sends).
- `SpeechRequest { text, voice: Option<Voice>, default_voice: Option<Voice>, format: Option<AudioFormat>, speed, language, instructions, name }`. `name` is never sent to the provider. `default_voice` is used when `voice` is `None`, and is the voice that speaks when `voice` turns out to be a description on a provider that needs a base voice (OpenAI). Fields an API has no equivalent for are not sent, and each provider's docs say which.
- `AudioFormat` (`#[non_exhaustive]`): `Mp3`, `Wav`, `Opus`, `Flac`, `Aac`, `Pcm`.
- `SpeechResponse { audio: Media, id: Option<String>, sample_rate: Option<u32> }`. `sample_rate` matters for headerless PCM, whose rate differs per provider (OpenAI 24 kHz; VoxCPM2 presumably 48 kHz); `None` when unknown.
- `CreateVoice { name, samples: Vec<Media>, transcript: Option<String>, description }`.
- `VoiceInfo { id, name, description, cloned: bool }`, `AudioModel` as `ImageModel`.
- `AudioError`: as `ImageError` (`Http`, `Api`, `InvalidResponse`, `Unsupported`, `UnknownProvider`, `MissingApiKey`, `Storage`, `is_auth()`) plus `MissingVoice` and `InvalidRequest(String)`.
- `from_model` / `list_models(provider)` via `ProviderKind::build`: `provider::model` prefix, or a well-known name (`tts-1*`, `gpt-4o-mini-tts*` → OpenAI; `eleven_*` → ElevenLabs).

### Providers

Bodies come back as raw bytes (`HttpResponse.body`). Multipart upload exists (`Multipart::with_file`). `send_request`/`api_error` in `agentai-image` are `pub(super)` and return `ImageError`, so copy the small helpers rather than sharing them.

| | OpenAI (`provider/openai.rs`) | ElevenLabs (`provider/elevenlabs.rs`) | oMLX (`provider/omlx.rs`) |
|---|---|---|---|
| Endpoint | `POST /audio/speech` | `POST /v1/text-to-speech/{voice_id}?output_format=` | `POST /audio/speech` |
| Env | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | `ELEVENLABS_API_KEY`, `ELEVENLABS_BASE_URL` | `OMLX_ENDPOINT`, `OMLX_API_KEY` |
| Auth | Bearer | `xi-api-key` | Bearer if a key is set |
| `Voice::Id` | preset string; `voice_...` as `{"id": ...}` | path segment | looked up in `VoiceStore`, sent as `ref_audio` + `ref_text` |
| `Voice::Description` | `default_voice` + `instructions` (`Unsupported` on `tts-1*`; `MissingVoice` without a default) | `Unsupported` | `instructions` |
| `Voice::Auto` | preset or `voice_...` → Id, otherwise Description | Id | in `VoiceStore` → Id, otherwise Description |
| No voice and no `default_voice` | `MissingVoice` | `MissingVoice` | omitted |
| `create_voice` | `Unsupported` | multipart `POST /v1/voices/add` (`name`, `files[]`, `description`) → `voice_id` | stores the sample in `VoiceStore`, returns its ID |
| `list_voices` | built-in presets (9 on `tts-1*`, 13 otherwise) | `GET /v2/voices`, following `next_page_token` | `VoiceStore` entries |
| `list_models` | from `GET /models`, official base URL only | `GET /v1/models`, keep `can_do_text_to_speech` | `Unsupported` |
| Formats | all | `AudioFormat` mapped to `output_format`; unmapped → `Unsupported` | all except `Aac` |

`supports_voice_description()`: OpenAI true except on `tts-1*`, ElevenLabs false, oMLX true.

ElevenLabs takes an escape hatch `with_output_format("mp3_44100_128")` for values `AudioFormat` cannot express.

oMLX notes:
- `ref_audio` is strict base64 with no `data:` prefix, maximum 20 MB encoded (return `InvalidRequest` above it).
- `ref_text` is required by oMLX even though VoxCPM2 ignores it. When `CreateVoice.transcript` is `None`, send a fixed placeholder so the shared `create_voice` call works; confirm the placeholder against a running server.
- More than one sample → `InvalidRequest`.
- Do not send `voice`: oMLX reroutes it to VoxCPM2's `instruct`. Use `instructions`.
- `speed`, `language` and samplers appear to have no effect on VoxCPM2; document it.
- Default response format is the server's `wav`.

### `VoiceStore`

`MediaStorage` cannot list, and a clone needs its transcript too, so a small trait lives in `agentai-audio`: `save(StoredVoice) -> id`, `get(id)`, `list()`, plus `MemoryVoiceStore`. The store is explicit only: `OmlxProvider::from_env()?.with_voice_store(store)`, then `SpeechGenerator::from_provider`. A provider built by `from_model` or `SpeechGenerator::new("omlx::...")` has no store and never creates one silently, so clones cannot vanish unnoticed. Without a store, `supports_voice_cloning()` is false, `create_voice`, `Voice::Id` and `list_voices` return `Unsupported`, and `Voice::Auto` is treated as a description. The docs say this next to `from_model`.

### `generator.rs`

`SpeechGenerator` as `ImageGenerator`: `new(model)`, `from_provider`, defaults (`with_default_voice`, which fills `SpeechRequest::default_voice`; `with_format`, `with_speed`, `with_language`), optional `MediaSave` that sets `SpeechResponse::id`. File name is `SpeechRequest::name` or `speech`, with the extension from a table (`audio/mpeg` → `.mp3`, not the MIME subtype as in `ImageGenerator::store`). Also `create_voice` and `list_voices` passthroughs.

### `toolbox.rs` (on `#[toolbox]`)

`SpeechToolBox::new(generator, storage: impl MediaStorage)`: the storage is required, so paid-for audio always ends up somewhere and the LLM always gets an ID.
- `generate_speech(text, name, voice: Option<String>, instructions: Option<String>)`. Without `with_voices`, `voice` becomes `Voice::auto`. With `with_voices`, the list is an allowlist: a value on the list becomes `Voice::id`; anything else never reaches the provider as an ID, it becomes `Voice::description` when `supports_voice_description()`, otherwise `ToolError::Failed`. Without `voice`, the generator's default applies. Recoverable provider failures (`MissingVoice`, `Unsupported`, API 4xx) are `ToolError::Failed` so the LLM can correct itself.
- `list_voices()`: JSON text with the voices (`with_voices(Vec<VoiceInfo>)`, otherwise the provider's list) and `descriptions_supported`.

Output `SpeechToolOutput::{Metadata, Media}`, **default `Metadata`** (differs from images on purpose): only some LLM APIs accept audio input, so the audio is not sent back unless the app asks.

No cloning tool. `ReadMediaToolBox` stays in `agentai-image`.

## Decided by Claude, not by the user (check these)

- Names `SpeechProvider`, `SpeechGenerator`, `SpeechToolBox`, `AudioError`, and the `Voice::Auto` variant.
- `VoiceStore` trait instead of extending `MediaStorage`.
- `SpeechToolBox` requires a storage, unlike `GenerateImageToolBox`.
- Tool output defaults to `Metadata`, unlike images.
- `default_voice` on the request and generator as the base voice for descriptions.
- OpenAI description mapped to `instructions`; oMLX placeholder `ref_text`.
- Deleting voices left out.

## Left to the builder

Internal module split, exact helper signatures, private wire structs, doc text. If the `#[toolbox]` macro blocks `Option<String>` parameters, explain the blocker and propose a macro change rather than hand-writing `ToolBox`.

## Critical files

New: `crates/agentai-audio/{Cargo.toml,README.md,src/{lib,generator,toolbox,voice_store}.rs,src/provider/{mod,openai,elevenlabs,omlx}.rs,examples/*}`.

Reuse: `agentai_core::http::{HttpClient, HttpRequest, HttpResponse, Multipart}`, `agentai_core::message::Media` (`from_bytes`), `agentai_core::storage`, `agentai_core::tls::install_crypto_provider`. Copy the test mock from `crates/agentai-image/src/provider/openai.rs` (`MockHttp`).

Register: `release-plz.toml` (version group); `crates/agentai/src/lib.rs` then regenerate README with cargo-rdme; `llms.txt`, `llms-full.txt`; `CLAUDE.md`; `crates/agentai/src/tool/mod.rs` crate list. No `MIGRATION.md` entry: nothing breaks. Root `Cargo.toml` and CI pick up `crates/*` automatically.

## Verification

- `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test -p agentai-audio`; `cargo tree -i aws-lc-rs` finds nothing.
- Unit tests per CLAUDE.md, on the mock client: request bodies per provider, `Voice::Auto` routing, `MissingVoice`, ElevenLabs multipart and `next_page_token`, oMLX `ref_audio`/`ref_text` from the store, extension table.
- Live, with the local `.env`: examples generate speech and create and use a clone on oMLX with `mlx-community/VoxCPM2-8bit`. Whether oMLX loads VoxCPM2 is unconfirmed (issue jundot/omlx#1305); test it first. If it fails, stop and report.
- ElevenLabs and OpenAI live runs need keys; say plainly if they were not run.

## After approval

Commit this design as `docs/superpowers/specs/2026-11-09-agentai-audio-design.md`, then write the implementation plan with `superpowers:writing-plans`.
