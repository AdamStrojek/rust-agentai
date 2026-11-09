//! Clones a voice from a WAV sample with a local oMLX server and speaks with it, saving the audio
//! in `audio/`.
//!
//! ```bash
//! # OMLX_ENDPOINT and OMLX_API_KEY point to the server
//! AGENTAI_SPEECH_MODEL=mlx-community/VoxCPM2-8bit \
//!     cargo run -p agentai-audio --example clone_voice -- sample.wav
//! ```
//!
//! oMLX has no voice library: it clones from the sample sent with every request, so the provider
//! keeps the sample in a `VoiceStore`. This one is `MemoryVoiceStore`, gone when the program ends.
//! Pass the transcript of the sample as the second argument to get a closer clone.

use agentai::{Media, MediaSource};
use agentai_audio::provider::omlx::OmlxProvider;
use agentai_audio::{CreateVoice, MemoryVoiceStore, SpeechGenerator, SpeechRequest, Voice};
use base64::Engine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let sample_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: clone_voice <sample.wav> [transcript]"))?;
    let transcript = args.next();
    let model = std::env::var("AGENTAI_SPEECH_MODEL")
        .unwrap_or_else(|_| "mlx-community/VoxCPM2-8bit".to_string());

    let provider = OmlxProvider::from_env(model)?.with_voice_store(MemoryVoiceStore::new());
    let generator = SpeechGenerator::from_provider(provider);

    let sample = Media::from_bytes("audio/wav", std::fs::read(&sample_path)?);
    let mut request = CreateVoice::new("sample", vec![sample]);
    if let Some(transcript) = transcript {
        request = request.with_transcript(transcript);
    }
    let voice_id = generator.create_voice(request).await?;
    println!("Cloned voice {voice_id}");

    let response = generator
        .generate_with(
            SpeechRequest::new("This is my cloned voice, speaking a sentence it never heard.")
                .with_voice(Voice::id(voice_id)),
        )
        .await?;

    match &response.audio.source {
        MediaSource::Base64(data) => {
            std::fs::create_dir_all("audio")?;
            let path = "audio/cloned.wav";
            std::fs::write(
                path,
                base64::engine::general_purpose::STANDARD.decode(data)?,
            )?;
            println!("Saved {path} ({})", response.audio.content_type);
        }
        MediaSource::Url(url) => println!("The audio is at {url}"),
    }
    Ok(())
}
