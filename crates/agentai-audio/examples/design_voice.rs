//! Designs a voice from a description, saves the previews in `audio/`, keeps the first one as a
//! voice and speaks with it.
//!
//! ```bash
//! # Fish Audio (FISH_API_KEY)
//! cargo run -p agentai-audio --example design_voice
//! # ElevenLabs (ELEVENLABS_API_KEY)
//! AGENTAI_SPEECH_PROVIDER=elevenlabs cargo run -p agentai-audio --example design_voice
//! # A local oMLX server (OMLX_ENDPOINT, OMLX_API_KEY), with a model that designs voices
//! AGENTAI_SPEECH_PROVIDER=omlx cargo run -p agentai-audio --example design_voice
//! ```
//!
//! - `AGENTAI_SPEECH_PROVIDER`: `fishaudio` (default), `elevenlabs` or `omlx`; the others cannot
//!   design voices.
//! - `AGENTAI_SPEECH_MODEL`: the speech model that speaks with the new voice (default `s2-pro`,
//!   `eleven_multilingual_v2` or `mlx-community/VoxCPM2-8bit`).
//! - `AGENTAI_DESIGN_MODEL`: the oMLX model that designs the voice (default: the speech model).
//!   Fish Audio designs with `voice-design-1` and ElevenLabs with `eleven_ttv_v3`.
//!
//! oMLX designs a voice by speaking the text with the description as instructions, and keeps a
//! chosen preview in a `VoiceStore`, so this example gives it a `MemoryVoiceStore` (gone when
//! the program ends).

use agentai::{Media, MediaSource};
use agentai_audio::{
    MemoryVoiceStore, SpeechGenerator, SpeechProviderKind, base64, speech_provider,
};

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
    let kind: SpeechProviderKind = std::env::var("AGENTAI_SPEECH_PROVIDER")
        .as_deref()
        .unwrap_or("fishaudio")
        .parse()?;
    let speech_model = |default: &str| {
        std::env::var("AGENTAI_SPEECH_MODEL").unwrap_or_else(|_| default.to_string())
    };
    let builder = match kind {
        SpeechProviderKind::FishAudio => {
            speech_provider(kind, speech_model("s2-pro")).with_design_model("voice-design-1")
        }
        SpeechProviderKind::ElevenLabs => {
            speech_provider(kind, speech_model("eleven_multilingual_v2"))
                .with_design_model("eleven_ttv_v3")
        }
        SpeechProviderKind::Omlx => {
            let model = speech_model("mlx-community/VoxCPM2-8bit");
            let design_model = std::env::var("AGENTAI_DESIGN_MODEL").unwrap_or(model.clone());
            speech_provider(kind, model)
                .with_design_model(design_model)
                .with_voice_store(MemoryVoiceStore::new())
        }
        other => anyhow::bail!("{other} cannot design voices"),
    };
    let generator = SpeechGenerator::new(builder.build()?);

    // The text is required by oMLX, optional for ElevenLabs and Fish Audio
    let previews = generator
        .design_voice(DESCRIPTION)
        .with_text(TEXT)
        .with_count(2)
        .await?;
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
        .generate("This sentence is spoken with the voice I designed.")
        .with_voice(voice_id)
        .await?;
    save(&response.audio, "designed")
}
