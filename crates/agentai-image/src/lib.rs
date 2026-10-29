//! # AgentAI Image
//!
//! Image generation and editing for [AgentAI](https://crates.io/crates/agentai): from your
//! code with [`ImageGenerator`], or by an agent with the toolboxes (feature `toolbox`).
//!
//! ## Generating images
//!
//! ```rust,no_run
//! use agentai_image::ImageGenerator;
//!
//! # async fn run() -> Result<(), agentai_image::ImageError> {
//! // The provider comes from the model name, the API key from `OPENAI_API_KEY`
//! let generator = ImageGenerator::new("openai::gpt-image-1")?.with_size("1024x1024");
//! let response = generator.generate("A lighthouse at dawn").await?;
//! // response.image is an agentai `Media`, base64 data or a URL
//! # Ok(())
//! # }
//! ```
//!
//! With [`with_storage`](ImageGenerator::with_storage) every image is saved in a
//! [`MediaSave`](agentai_core::storage::MediaSave) and gets an ID.
//!
//! ## Tools for agents
//!
//! - [`GenerateImageToolBox`]: `generate_image`, and `edit_image` with a storage. The model
//!   gets the image, or only its metadata as JSON ([`ImageToolOutput`]).
//! - [`ReadMediaToolBox`]: `read_media` gives the model a stored file by ID.
//!
//! ```rust,no_run
//! # #[cfg(feature = "toolbox")]
//! # async fn run() -> anyhow::Result<()> {
//! use agentai::Agent;
//! use agentai::tool::ToolBoxSet;
//! use agentai_core::storage::MemoryStorage;
//! use agentai_image::{GenerateImageToolBox, ImageGenerator, ReadMediaToolBox};
//!
//! let storage = MemoryStorage::new();
//! let mut tools = ToolBoxSet::new();
//! tools.add_tool(
//!     GenerateImageToolBox::new(ImageGenerator::new("openai::gpt-image-1")?)
//!         .with_storage(storage.clone()),
//! );
//! tools.add_tool(ReadMediaToolBox::new(storage.clone()));
//! let agent = Agent::new("claude-haiku-5-5", "You are an illustrator.")?.with_toolbox(tools);
//! let result = agent.run::<String>("Draw a lighthouse at dawn").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Providers
//!
//! The image APIs live in [`provider`]. [`ImageGenerator::new`] picks one from the model name
//! (`openai::gpt-image-1`) with its settings from environment variables. To set the API key
//! or base URL in code, build the provider and pass it to [`ImageGenerator::from_provider`]:
//!
//! ```rust,no_run
//! use agentai_image::ImageGenerator;
//! use agentai_image::provider::openai::OpenAiImageProvider;
//!
//! let provider = OpenAiImageProvider::new("gpt-image-1", "sk-...", "https://api.openai.com/v1");
//! let generator = ImageGenerator::from_provider(provider);
//! ```
//!
//! Implement [`provider::ImageProvider`] to use another API.
//!
//! ## Features
#![doc = document_features::document_features!()]

mod generator;
pub mod provider;
#[cfg(feature = "toolbox")]
mod toolbox;

pub use generator::ImageGenerator;
pub use provider::{ImageError, ImageResponse};
#[cfg(feature = "toolbox")]
pub use toolbox::{GenerateImageToolBox, ImageToolOutput, ReadMediaToolBox};
