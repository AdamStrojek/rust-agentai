# agentai-audio

Text-to-speech and voice cloning for [AgentAI](https://crates.io/crates/agentai).

- `SpeechProvider` is the interface to a speech API; each provider lives in `agentai_audio::provider`.
- Audio is returned as an `agentai` `Media` and can be saved in any `MediaStorage` from `agentai_audio::storage`, re-exported from `agentai-core`.
- TLS features as in `agentai`: `rustls-ring` (default), `rustls-aws-lc`, `native-tls`, `system-proxy`.
