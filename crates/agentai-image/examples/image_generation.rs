//! An agent that draws an image and then edits it. The images are saved in `images/`.
//!
//! ```bash
//! OPENAI_API_KEY=<your key> cargo run -p agentai-image --example image_generation
//! ```
//!
//! The agent's model comes from `AGENTAI_MODEL` (default `claude-haiku-5-5`).

use agentai::tool::ToolError;
use agentai::{Agent, Media, MediaSource};
use agentai_image::{ImageResponse, ImageSink, ImageSource, ImageToolBox};
use async_trait::async_trait;
use base64::Engine;
use std::path::PathBuf;

/// Saves the images in a directory and loads them from it by name.
#[derive(Clone)]
struct ImageDir(PathBuf);

#[async_trait]
impl ImageSink for ImageDir {
    async fn save(&self, response: &ImageResponse) {
        let image = &response.image;
        let name = image.name.clone().unwrap_or_else(|| "image".to_string());
        let bytes = match &image.source {
            MediaSource::Base64(data) => {
                match base64::engine::general_purpose::STANDARD.decode(data) {
                    Ok(bytes) => bytes,
                    Err(err) => return eprintln!("Cannot decode {name}: {err}"),
                }
            }
            MediaSource::Url(url) => return println!("{name} is at {url}"),
            _ => return,
        };
        let path = self.0.join(&name);
        match std::fs::write(&path, bytes) {
            Ok(()) => println!("Saved {}", path.display()),
            Err(err) => eprintln!("Cannot save {}: {err}", path.display()),
        }
    }
}

#[async_trait]
impl ImageSource for ImageDir {
    async fn load(&self, reference: &str) -> Result<Media, ToolError> {
        // Only names of files in the directory
        if reference.contains(['/', '\\']) || reference.starts_with('.') {
            return Err(ToolError::InvalidArguments(format!(
                "'{reference}' is not an image name"
            )));
        }
        let bytes = std::fs::read(self.0.join(reference)).map_err(|err| {
            ToolError::InvalidArguments(format!("cannot read image '{reference}': {err}"))
        })?;
        let content_type = match reference.rsplit('.').next() {
            Some("jpeg" | "jpg") => "image/jpeg",
            Some("webp") => "image/webp",
            _ => "image/png",
        };
        Ok(Media::from_bytes(content_type, bytes).with_name(reference))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::var("AGENTAI_MODEL").unwrap_or_else(|_| "claude-haiku-5-5".to_string());
    let dir = ImageDir(PathBuf::from("images"));
    std::fs::create_dir_all(&dir.0)?;

    let images = ImageToolBox::new("openai::gpt-image-1")?
        .with_size("1024x1024")
        .with_quality("low")
        .with_sink(dir.clone())
        .with_source(dir);
    let agent = Agent::new(
        &model,
        "You are an illustrator. Use your tools to draw what the user asks for.",
    )?
    .with_toolbox(images);

    let result = agent
        .run::<String>(
            "Draw a small lighthouse on a rock at dawn, then edit that image to add a sailing boat.",
        )
        .await?;
    println!("{}", result.answer);
    Ok(())
}
