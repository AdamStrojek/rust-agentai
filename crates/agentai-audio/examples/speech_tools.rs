//! An agent that speaks texts with `generate_speech`, with the audio kept in `audio/`.
//!
//! ```bash
//! OPENAI_API_KEY=<your key> cargo run -p agentai-audio --example speech_tools
//! # A local oMLX server (OMLX_ENDPOINT, OMLX_API_KEY)
//! AGENTAI_SPEECH_MODEL=omlx::mlx-community/VoxCPM2-8bit \
//!     cargo run -p agentai-audio --example speech_tools
//! ```
//!
//! The agent's model comes from `AGENTAI_MODEL` (default `claude-haiku-5-5`); it does not need to
//! hear audio, because `generate_speech` answers with metadata (ID, type, size) only. The voice
//! tools never clone voices: `create_voice` stays in your code.

use agentai::tool::ToolBoxSet;
use agentai::{Agent, Media, MediaSource};
use agentai_audio::storage::{MediaRead, MediaSave, StorageError};
use agentai_audio::{SpeechGenerator, SpeechToolBox};
use async_trait::async_trait;
use base64::Engine;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Keeps media as files in a directory; the ID is the file name.
struct DirStorage {
    dir: PathBuf,
    count: AtomicUsize,
}

#[async_trait]
impl MediaSave for DirStorage {
    async fn save(&self, media: &Media) -> Result<String, StorageError> {
        let bytes = match &media.source {
            MediaSource::Base64(data) => base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(StorageError::other)?,
            MediaSource::Url(_) => return Err(StorageError::other("only stores bytes")),
        };
        let number = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        let name = media.name.as_deref().unwrap_or("speech");
        // Only a file name, whatever the model wrote
        let name = name.replace(['/', '\\'], "_");
        let id = format!("{number}-{}", name.trim_start_matches('.'));
        std::fs::write(self.dir.join(&id), bytes).map_err(StorageError::other)?;
        println!("Saved {}", self.dir.join(&id).display());
        Ok(id)
    }
}

#[async_trait]
impl MediaRead for DirStorage {
    async fn read(&self, id: &str) -> Result<Media, StorageError> {
        if id.contains(['/', '\\']) || id.starts_with('.') {
            return Err(StorageError::NotFound(id.to_string()));
        }
        let bytes =
            std::fs::read(self.dir.join(id)).map_err(|_| StorageError::NotFound(id.to_string()))?;
        let content_type = match id.rsplit('.').next() {
            Some("mp3") => "audio/mpeg",
            Some("flac") => "audio/flac",
            Some("ogg" | "opus") => "audio/ogg",
            _ => "audio/wav",
        };
        Ok(Media::from_bytes(content_type, bytes).with_name(id))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("AGENTAI_MODEL").unwrap_or_else(|_| "claude-haiku-5-5".to_string());
    let speech_model = std::env::var("AGENTAI_SPEECH_MODEL")
        .unwrap_or_else(|_| "openai::gpt-4o-mini-tts".to_string());
    std::fs::create_dir_all("audio")?;
    let storage = Arc::new(DirStorage {
        dir: PathBuf::from("audio"),
        count: AtomicUsize::new(0),
    });

    let generator = SpeechGenerator::new(&speech_model)?;
    let mut tools = ToolBoxSet::new();
    tools.add_tool(SpeechToolBox::new(generator, storage));
    let agent = Agent::new(
        &model,
        "You are a voice assistant. Use your tools to speak the texts the user asks for.",
    )?
    .with_toolbox(tools);

    let result = agent
        .run::<String>(
            "Say 'Welcome aboard, the show starts in five minutes' in a calm voice and name the file welcome. \
             Then tell me the ID of the audio file.",
        )
        .await?;
    println!("{}", result.answer);
    Ok(())
}
