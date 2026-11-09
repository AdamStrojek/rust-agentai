# agentai-audio

Text-to-speech and voice cloning for [AgentAI](https://crates.io/crates/agentai).

- `SpeechGenerator` speaks texts and clones voices from your code. OpenAI and ElevenLabs need a voice
  (a default one or one per request), or the request fails with `MissingVoice`.
- With the `toolbox` feature (on by default), `SpeechToolBox` gives an agent the `generate_speech` and
  `list_voices` tools. The model gets only the metadata of the audio as JSON by default
  (`SpeechToolOutput::Metadata`), or the audio itself (`SpeechToolOutput::Media`); `with_voices` limits
  the voices to an allowlist. No tool clones voices.
- Providers in `agentai_audio::provider`: OpenAI and compatible servers (OpenAI shuts these speech models
  down on 2027-01-06), ElevenLabs and oMLX. oMLX clones voices from a sample kept in a `VoiceStore`
  that you give it explicitly (`MemoryVoiceStore`).
- Audio is returned as an `agentai` `Media` and can be saved in any `MediaStorage` from `agentai_audio::storage`, re-exported from `agentai-core`.
- TLS features as in `agentai`: `rustls-ring` (default), `rustls-aws-lc`, `native-tls`, `system-proxy`.

```rust,ignore
use agentai_audio::{SpeechGenerator, Voice};

// The provider comes from the model name, the API key from `OPENAI_API_KEY`
let generator = SpeechGenerator::new("openai::gpt-4o-mini-tts")?
    .with_default_voice(Voice::id("alloy"));
let response = generator.generate("Good morning").await?;
```

See the `speech_generator`, `speech_tools` and `clone_voice` examples.
