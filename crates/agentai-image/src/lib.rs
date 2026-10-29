//! # AgentAI Image
//!
//! Image generation and editing tools for [AgentAI](https://crates.io/crates/agentai) agents.
//!
//! [`ImageToolBox`] is a [`ToolBox`](agentai::tool::ToolBox) with the tools:
//! - `generate_image`: draws an image from a prompt the model writes.
//! - `edit_image`: changes images the model refers to by name. Offered when you give the
//!   toolbox an [`ImageSource`] that loads them.
//!
//! The image goes back to the model as the tool result, as [`Media`](agentai::Media), so it
//! is also in [`AgentResult::new_messages`](agentai::AgentResult::new_messages). To store
//! every image as it is made (a directory, your UI, ...), give the toolbox an [`ImageSink`].
//!
//! ```rust,no_run
//! use agentai::Agent;
//! use agentai_image::ImageToolBox;
//!
//! # async fn run() -> anyhow::Result<()> {
//! // The provider comes from the model name, the API key from `OPENAI_API_KEY`
//! let images = ImageToolBox::new("openai::gpt-image-1")?;
//! let agent = Agent::new("claude-haiku-5-5", "You are an illustrator.")?.with_toolbox(images);
//! let result = agent.run::<String>("Draw a lighthouse at dawn").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Providers
//!
//! The image APIs live in [`provider`]. [`ImageToolBox::new`] picks one from the model name
//! (`openai::gpt-image-1`) with its settings from environment variables. To set the API key
//! or base URL in code, build the provider and pass it to [`ImageToolBox::from_provider`]:
//!
//! ```rust,no_run
//! use agentai_image::ImageToolBox;
//! use agentai_image::provider::openai::OpenAiImageProvider;
//!
//! let provider = OpenAiImageProvider::new("gpt-image-1", "sk-...", "https://api.openai.com/v1");
//! let images = ImageToolBox::from_provider(provider);
//! ```
//!
//! Implement [`provider::ImageProvider`] to use another API.
//!
//! See the `image_generation` example for a sink and a source that keep the images in a
//! directory.

pub mod provider;
mod toolbox;

pub use provider::{ImageError, ImageResponse};
pub use toolbox::{ImageSink, ImageSource, ImageToolBox};
