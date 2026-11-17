//! Speaks a dialogue of two voices with Fish Audio `drama-3`, saving the audio in `audio/`.
//!
//! ```bash
//! # FISH_API_KEY is the Fish Audio key
//! cargo run -p agentai-audio --example dialogue -- \
//!     ala.wav "What Ala says in her sample." ola.wav "What Ola says in her sample."
//! ```
//!
//! The samples go to a `VoiceStore`, so they are sent with the request; the ID of a voice model
//! in the Fish Audio account works the same way in `with_voices`. The order of the voices is the
//! speaker index used by the `<|speaker:N|>` tags in the text. The time of every word is printed.

use agentai::{Media, MediaSource};
use agentai_audio::provider::fishaudio::FishAudioProvider;
use agentai_audio::{
    MemoryVoiceStore, SpeechGenerator, SpeechRequest, StoredVoice, Voice, VoiceStore,
};
use base64::Engine;

const USAGE: &str = "usage: dialogue <a.wav> <transcript a> <b.wav> <transcript b>";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [sample_a, transcript_a, sample_b, transcript_b] = args.as_slice() else {
        anyhow::bail!(USAGE);
    };

    let store = MemoryVoiceStore::new();
    let mut voices = Vec::new();
    for (name, path, transcript) in [("a", sample_a, transcript_a), ("b", sample_b, transcript_b)] {
        let sample = Media::from_bytes("audio/wav", std::fs::read(path)?);
        let id = store
            .save(StoredVoice::new(name, sample).with_transcript(transcript))
            .await?;
        voices.push(Voice::id(id));
    }

    let provider = FishAudioProvider::from_env("drama-3")?.with_voice_store(store);
    let generator = SpeechGenerator::from_provider(provider);
    let response = generator
        .generate_with(
            SpeechRequest::new(
                "<|speaker:0|>Did you hear that?<|speaker:1|>Hear what? It is quiet.\
                 <|speaker:0|>Exactly. Too quiet.",
            )
            .with_voices(voices)
            .with_timestamps(true),
        )
        .await?;
    for word in response.timestamps.iter().flatten() {
        println!("{:>6.2}-{:>6.2}s {}", word.start, word.end, word.text);
    }

    match &response.audio.source {
        MediaSource::Base64(data) => {
            std::fs::create_dir_all("audio")?;
            let path = "audio/dialogue.mp3";
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
