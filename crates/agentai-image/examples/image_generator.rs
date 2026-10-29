//! Generates an image with `ImageGenerator`, without an agent, edits it, and saves both in
//! `images/`.
//!
//! ```bash
//! OPENAI_API_KEY=<your key> cargo run -p agentai-image --example image_generator
//! # A local OpenAI-compatible server
//! OPENAI_BASE_URL=http://localhost:3000/v1 AGENTAI_IMAGE_MODEL=openai::qwen_image_2.1 \
//!     cargo run -p agentai-image --example image_generator
//! ```

use agentai_core::message::MediaSource;
use agentai_image::{ImageGenerator, ImageResponse};
use base64::Engine;

fn save(response: &ImageResponse, name: &str) -> anyhow::Result<()> {
    if let Some(prompt) = &response.revised_prompt {
        println!("Prompt used: {prompt}");
    }
    match &response.image.source {
        MediaSource::Base64(data) => {
            std::fs::create_dir_all("images")?;
            let extension = response
                .image
                .content_type
                .rsplit('/')
                .next()
                .unwrap_or("png");
            let path = format!("images/{name}.{extension}");
            std::fs::write(
                &path,
                base64::engine::general_purpose::STANDARD.decode(data)?,
            )?;
            println!("Saved {path}");
        }
        MediaSource::Url(url) => println!("The image is at {url}"),
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model =
        std::env::var("AGENTAI_IMAGE_MODEL").unwrap_or_else(|_| "openai::gpt-image-1".to_string());
    let generator = ImageGenerator::new(&model)?.with_size("1024x1024");

    let response = generator
        .generate("A small lighthouse on a rock at dawn, watercolor")
        .await?;

    save(&response, "lighthouse")?;

    if generator.supports_edit() {
        let edited = generator
            .edit("Add a small sailing boat on the sea", vec![response.image])
            .await?;
        save(&edited, "lighthouse-with-boat")?;
    }
    Ok(())
}
