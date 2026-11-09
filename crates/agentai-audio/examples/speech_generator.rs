//! Speaks a text with `SpeechGenerator`, without an agent, and saves the audio in `audio/`.
//!
//! ```bash
//! OPENAI_API_KEY=<your key> AGENTAI_SPEECH_VOICE=alloy cargo run -p agentai-audio --example speech_generator
//! # A local oMLX server (OMLX_ENDPOINT, OMLX_API_KEY)
//! AGENTAI_SPEECH_MODEL=omlx::mlx-community/VoxCPM2-8bit \
//!     cargo run -p agentai-audio --example speech_generator
//! # ElevenLabs, with a voice ID from its voice library
//! ELEVENLABS_API_KEY=<your key> AGENTAI_SPEECH_MODEL=elevenlabs::eleven_multilingual_v2 \
//!     AGENTAI_SPEECH_VOICE=<voice ID> cargo run -p agentai-audio --example speech_generator
//! ```
//!
//! `AGENTAI_SPEECH_VOICE` is the default voice ID. OpenAI and ElevenLabs need one (without it the
//! request fails with `MissingVoice`); oMLX does not, and uses its own voice when it is unset.
//! With `omlx::`, setting it fails with `Unsupported`: `from_model` gives no voice store, so
//! there are no voice IDs.

use agentai::MediaSource;
use agentai_audio::{SpeechGenerator, Voice};
use base64::Engine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("AGENTAI_SPEECH_MODEL")
        .unwrap_or_else(|_| "openai::gpt-4o-mini-tts".to_string());
    let mut generator = SpeechGenerator::new(&model)?;
    if let Ok(voice) = std::env::var("AGENTAI_SPEECH_VOICE") {
        generator = generator.with_default_voice(Voice::id(voice));
    }

    let response = generator
        .generate("Good morning! This voice was made by an agent library written in Rust.")
        .await?;

    match &response.audio.source {
        MediaSource::Base64(data) => {
            std::fs::create_dir_all("audio")?;
            let name = response.audio.name.as_deref().unwrap_or("greeting.wav");
            let path = format!("audio/{name}");
            std::fs::write(
                &path,
                base64::engine::general_purpose::STANDARD.decode(data)?,
            )?;
            println!("Saved {path} ({})", response.audio.content_type);
        }
        MediaSource::Url(url) => println!("The audio is at {url}"),
    }
    Ok(())
}
