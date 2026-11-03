//! Lists the image models of a provider, with names ready for `ImageGenerator::new`.
//!
//! ```bash
//! # A local OpenAI-compatible server
//! OPENAI_BASE_URL=http://localhost:3000/v1 cargo run -p agentai-image --example list_models openai
//! # OpenRouter
//! OPENROUTER_API_KEY=<your key> cargo run -p agentai-image --example list_models openrouter
//! ```

use agentai_image::provider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let name = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("AGENTAI_IMAGE_PROVIDER").ok())
        .unwrap_or_else(|| "openai".to_string());

    for model in provider::list_models(&name).await? {
        let edits = match model.accepts_images {
            Some(true) => " (accepts images)",
            _ => "",
        };
        match model.name {
            Some(display) => println!("{} - {display}{edits}", model.model),
            None => println!("{}{edits}", model.model),
        }
    }
    Ok(())
}
