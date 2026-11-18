//! Designs a voice from a description, saves the previews in `audio/`, keeps the first one as a
//! voice and speaks with it.
//!
//! ```bash
//! # Fish Audio (FISH_API_KEY)
//! cargo run -p agentai-audio --example design_voice
//! # ElevenLabs (ELEVENLABS_API_KEY)
//! AGENTAI_SPEECH_MODEL=elevenlabs::eleven_multilingual_v2 \
//!     cargo run -p agentai-audio --example design_voice
//! # A local oMLX server (OMLX_ENDPOINT, OMLX_API_KEY), with a model that designs voices
//! AGENTAI_SPEECH_MODEL=omlx::mlx-community/VoxCPM2-8bit \
//!     cargo run -p agentai-audio --example design_voice
//! ```
//!
//! `AGENTAI_SPEECH_MODEL` defaults to `fishaudio::s2-pro`. oMLX designs a voice by speaking the
//! text with the description as instructions, and keeps a chosen preview in a `VoiceStore`, so
//! this example builds that provider with a `MemoryVoiceStore` (gone when the program ends)
//! instead of going through the model name.

use agentai::{Media, MediaSource};
use agentai_audio::provider::omlx::OmlxProvider;
use agentai_audio::{MemoryVoiceStore, SpeechGenerator, SpeechRequest, Voice, base64};

const DESCRIPTION: &str = "A warm, calm middle-aged woman with a low voice, speaking slowly";
const TEXT: &str = "Welcome back, and thank you for listening. Let me tell you how this voice was made, and why it sounds the way it does.";

fn save(media: &Media, name: &str) -> anyhow::Result<()> {
    match &media.source {
        MediaSource::Base64(data) => {
            std::fs::create_dir_all("audio")?;
            let path = format!("audio/{}", media.file_name(name));
            std::fs::write(&path, base64::decode(data)?)?;
            println!("Saved {path} ({})", media.content_type);
        }
        MediaSource::Url(url) => println!("The audio is at {url}"),
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model =
        std::env::var("AGENTAI_SPEECH_MODEL").unwrap_or_else(|_| "fishaudio::s2-pro".to_string());
    let generator = match model.strip_prefix("omlx::") {
        Some(model) => SpeechGenerator::from_provider(
            OmlxProvider::from_env(model)?.with_voice_store(MemoryVoiceStore::new()),
        ),
        None => SpeechGenerator::new(&model)?,
    };

    // The text is required by oMLX, optional for ElevenLabs and Fish Audio
    let previews = generator.design_voice(DESCRIPTION, Some(TEXT), 2).await?;
    for (n, preview) in previews.iter().enumerate() {
        save(&preview.audio, &format!("preview-{n}"))?;
    }
    let first = previews
        .first()
        .ok_or_else(|| anyhow::anyhow!("the provider returned no previews"))?;

    // The description is required by ElevenLabs to save a preview
    let voice_id = generator
        .create_voice("designed", first, Some(DESCRIPTION))
        .await?;
    println!("Saved the first preview as voice {voice_id}");

    let response = generator
        .generate_with(
            SpeechRequest::new("This sentence is spoken with the voice I designed.")
                .with_voice(Voice::id(voice_id)),
        )
        .await?;
    save(&response.audio, "designed")
}
