//! Speaks a dialogue of two voices with Fish Audio `drama-3`, saving the audio in `audio/`.
//!
//! ```bash
//! # FISH_API_KEY is the Fish Audio key
//! cargo run -p agentai-audio --example dialogue -- \
//!     ala.wav "What Ala says in her sample." ola.wav "What Ola says in her sample."
//! ```
//!
//! The provider speaks single texts with its speech model (`s2-pro`) and dialogues with its
//! dialogue model (`drama-3`, set with `with_dialogue_model`). The samples go to a `VoiceStore`
//! and every `DialogueTurn` names the voice that speaks it by its store ID, so the samples are
//! sent with the request. The IDs of voice models in your Fish Audio account work the same way,
//! but a dialogue takes either store IDs or account voices, not both. The text of a turn is plain
//! text: the provider marks the speakers itself. The time of every word is printed.

use agentai::{Media, MediaSource};
use agentai_audio::{
    DialogueTurn, MemoryVoiceStore, SpeechGenerator, StoredVoice, VoiceStore, base64,
    speech_provider,
};

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
        voices.push(id);
    }

    let provider = speech_provider("fishaudio", "s2-pro")
        .with_dialogue_model("drama-3")
        .with_voice_store(store)
        .build()?;
    let generator = SpeechGenerator::new(provider);
    let response = generator
        .generate_dialogue([
            DialogueTurn::new(&voices[0], "Did you hear that?"),
            DialogueTurn::new(&voices[1], "Hear what? It is quiet."),
            DialogueTurn::new(&voices[0], "Exactly. Too quiet."),
        ])
        .with_timestamps(true)
        .await?;
    for word in response.timestamps.iter().flatten() {
        println!("{:>6.2}-{:>6.2}s {}", word.start, word.end, word.text);
    }

    match &response.audio.source {
        MediaSource::Base64(data) => {
            std::fs::create_dir_all("audio")?;
            let path = format!("audio/{}", response.audio.file_name("dialogue"));
            std::fs::write(&path, base64::decode(data)?)?;
            println!("Saved {path} ({})", response.audio.content_type);
        }
        MediaSource::Url(url) => println!("The audio is at {url}"),
    }
    Ok(())
}
