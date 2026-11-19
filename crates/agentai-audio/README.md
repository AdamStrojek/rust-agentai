# agentai-audio

Text-to-speech, dialogues, voice cloning and voice design for [AgentAI](https://crates.io/crates/agentai).

- `speech_provider("fishaudio", "s2-pro")` (or `SpeechProviderKind::FishAudio.builder("s2-pro")`) builds a
  provider with its settings from environment variables or the builder (`with_api_key`, `with_base_url`).
  A provider has up to three engines, each with its own model: speech, dialogue (`with_dialogue_model`) and
  voice design (`with_design_model`). An engine without a model is off and returns `MissingModel`.
  `SpeechProviderKind::known_models()` lists the models this crate knows and what each can do.
- `SpeechGenerator` speaks texts, dialogues, clones voices from samples and designs voices from a
  description, from your code. Every call is a builder you `.await`:
  `generate(text).with_voice(id)`, `generate_dialogue([DialogueTurn::new(voice, text), ..])`,
  `design_voice(description).with_text(..).with_count(n)`. A voice is a `VoiceId` (a string). OpenAI and
  ElevenLabs need a voice, or the request fails with `MissingVoice`.
- Dialogues: Fish Audio (`drama-3`, `s2-pro`, ...) and ElevenLabs (`eleven_v3`, `eleven_v4`). The text of a
  turn is plain text; the provider marks the speakers.
- With the `toolbox` feature (on by default), `SpeechToolBox` gives an agent the `generate_speech` and
  `list_voices` tools. The model gets only the metadata of the audio as JSON by default
  (`SpeechToolOutput::Metadata`), or the audio itself (`SpeechToolOutput::Media`); `with_voices` limits
  the voices to an allowlist. No tool speaks dialogues, clones or designs voices.
- Providers in `agentai_audio::provider`: OpenAI and compatible servers (OpenAI shuts these speech models
  down on 2027-01-06), ElevenLabs, Fish Audio and oMLX. oMLX clones voices from a sample kept in a
  `VoiceStore` that you give it with `with_voice_store` (`MemoryVoiceStore`). Fish Audio sends the voices
  of a request either as samples from such a store (all voices in it) or as voice models of the account
  (none in it); a mix is an `InvalidRequest`.
- `with_timestamps(true)` returns the time of every word in `SpeechResponse::timestamps` (Fish Audio,
  through its timestamp stream; other providers return `Unsupported`).
- Audio is returned as an `agentai` `Media` and can be saved in any `MediaStorage` from
  `agentai_audio::storage`, re-exported from `agentai-core`.
- TLS features as in `agentai`: `rustls-ring` (default), `rustls-aws-lc`, `native-tls`, `system-proxy`.

```rust,ignore
use agentai_audio::{DialogueTurn, SpeechGenerator, speech_provider};

// The API key from `FISH_API_KEY`
let provider = speech_provider("fishaudio", "s2-pro")
    .with_dialogue_model("drama-3")
    .build()?;
let generator = SpeechGenerator::new(provider);
let response = generator.generate("Good morning").await?;
let dialogue = generator
    .generate_dialogue([
        DialogueTurn::new("<voice model ID>", "Ready?"),
        DialogueTurn::new("<other voice model ID>", "Always."),
    ])
    .await?;
```

See the `speech_generator`, `speech_tools`, `clone_voice`, `design_voice` and `dialogue` examples.
