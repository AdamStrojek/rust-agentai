//! Speaks a text with `SpeechGenerator`, without an agent, and saves the audio in `audio/`.
//!
//! ```bash
//! OPENAI_API_KEY=<your key> AGENTAI_SPEECH_VOICE=alloy \
//!     cargo run -p agentai-audio --example speech_generator
//! # ElevenLabs, with a voice ID from its voice library
//! ELEVENLABS_API_KEY=<your key> AGENTAI_SPEECH_PROVIDER=elevenlabs \
//!     AGENTAI_SPEECH_MODEL=eleven_multilingual_v2 AGENTAI_SPEECH_VOICE=<voice ID> \
//!     cargo run -p agentai-audio --example speech_generator
//! # A local oMLX server (OMLX_ENDPOINT, OMLX_API_KEY)
//! AGENTAI_SPEECH_PROVIDER=omlx AGENTAI_SPEECH_MODEL=mlx-community/VoxCPM2-8bit \
//!     cargo run -p agentai-audio --example speech_generator
//! ```
//!
//! - `AGENTAI_SPEECH_PROVIDER`: `openai` (default), `elevenlabs`, `fishaudio` or `omlx`.
//! - `AGENTAI_SPEECH_MODEL`: the speech model (default `gpt-4o-mini-tts`; set it with any other
//!   provider).
//! - `AGENTAI_SPEECH_VOICE`: the voice ID. OpenAI and ElevenLabs need one (without it the request
//!   fails with `MissingVoice`); Fish Audio and oMLX use their own voice when it is unset. This
//!   example builds oMLX without a voice store, so it has no voice IDs.

use agentai::MediaSource;
use agentai_audio::{SpeechGenerator, base64, speech_provider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider =
        std::env::var("AGENTAI_SPEECH_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    let model =
        std::env::var("AGENTAI_SPEECH_MODEL").unwrap_or_else(|_| "gpt-4o-mini-tts".to_string());
    let generator = SpeechGenerator::new(speech_provider(provider, model).build()?);

    let mut request = generator
        .generate("Good morning! This voice was made by an agent library written in Rust.")
        .with_name("greeting");
    if let Ok(voice) = std::env::var("AGENTAI_SPEECH_VOICE") {
        request = request.with_voice(voice);
    }
    let response = request.await?;

    match &response.audio.source {
        MediaSource::Base64(data) => {
            std::fs::create_dir_all("audio")?;
            let path = format!("audio/{}", response.audio.file_name("greeting"));
            std::fs::write(&path, base64::decode(data)?)?;
            println!("Saved {path} ({})", response.audio.content_type);
        }
        MediaSource::Url(url) => println!("The audio is at {url}"),
    }
    Ok(())
}
