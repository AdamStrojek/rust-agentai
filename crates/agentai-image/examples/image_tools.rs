//! An agent that draws an image and then edits it, with the images kept in `images/`.
//!
//! ```bash
//! OPENAI_API_KEY=<your key> cargo run -p agentai-image --example image_tools
//! # A local OpenAI-compatible server
//! OPENAI_BASE_URL=http://localhost:3000/v1 AGENTAI_IMAGE_MODEL=openai::qwen_image_2.1 \
//!     cargo run -p agentai-image --example image_tools
//! ```
//!
//! The agent's model comes from `AGENTAI_MODEL` (default `claude-haiku-5-5`). The image tools
//! answer with metadata only, and the model looks at an image with `read_media` when it needs to.

use agentai::Agent;
use agentai::tool::ToolBoxSet;
use agentai_core::message::{Media, MediaSource};
use agentai_core::storage::{MediaRead, MediaSave, StorageError};
use agentai_image::{GenerateImageToolBox, ImageGenerator, ImageToolOutput, ReadMediaToolBox};
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
        let id = format!("{number}-{}", media.name.as_deref().unwrap_or("file"));
        std::fs::write(self.dir.join(&id), bytes).map_err(StorageError::other)?;
        println!("Saved {}", self.dir.join(&id).display());
        Ok(id)
    }
}

#[async_trait]
impl MediaRead for DirStorage {
    async fn read(&self, id: &str) -> Result<Media, StorageError> {
        // Only names of files in the directory
        if id.contains(['/', '\\']) || id.starts_with('.') {
            return Err(StorageError::NotFound(id.to_string()));
        }
        let bytes =
            std::fs::read(self.dir.join(id)).map_err(|_| StorageError::NotFound(id.to_string()))?;
        let content_type = match id.rsplit('.').next() {
            Some("jpeg" | "jpg") => "image/jpeg",
            Some("webp") => "image/webp",
            _ => "image/png",
        };
        Ok(Media::from_bytes(content_type, bytes).with_name(id))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("AGENTAI_MODEL").unwrap_or_else(|_| "claude-haiku-5-5".to_string());
    let image_model =
        std::env::var("AGENTAI_IMAGE_MODEL").unwrap_or_else(|_| "openai::gpt-image-1".to_string());
    std::fs::create_dir_all("images")?;
    let storage = Arc::new(DirStorage {
        dir: PathBuf::from("images"),
        count: AtomicUsize::new(0),
    });

    let generator = ImageGenerator::new(&image_model)?.with_size("1024x1024");
    let mut tools = ToolBoxSet::new();
    tools.add_tool(
        GenerateImageToolBox::new(generator)
            .with_storage(storage.clone())
            .with_output(ImageToolOutput::Metadata),
    );
    tools.add_tool(ReadMediaToolBox::new(storage));
    let agent = Agent::new(
        &model,
        "You are an illustrator. Use your tools to draw what the user asks for.",
    )?
    .with_toolbox(tools);

    let result = agent
        .run::<String>(
            "Draw a small lighthouse on a rock at dawn, then edit that image to add a sailing boat.",
        )
        .await?;
    println!("{}", result.answer);
    Ok(())
}
