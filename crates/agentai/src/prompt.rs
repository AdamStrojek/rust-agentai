//! # Prompts
//!
//! A [`Prompt`] is what you send to the model in a run: text, images or other media, in order.
//! [`Agent::run`](crate::Agent::run) and [`Agent::run_stream`](crate::Agent::run_stream) take
//! anything that implements [`IntoPrompt`], so plain text works as before:
//!
//! ```no_run
//! use agentai::{Agent, AgentResult, Media, Prompt};
//!
//! # async fn example(agent: Agent, png: Vec<u8>) -> Result<(), agentai::AgentError> {
//! let result: AgentResult<String> = agent.run("Why is the sky blue?").await?;
//!
//! // Vision models also accept images
//! let prompt = Prompt::text("What is in this picture?")
//!     .with_image_url("https://example.com/cat.jpg")
//!     .with_media(Media::from_bytes("image/png", &png));
//! let result: AgentResult<String> = agent.run(prompt).await?;
//! # Ok(())
//! # }
//! ```

use crate::message::{Media, Message, Part, Role};

/// Content of the user message sent in a run: text and media, in order.
///
/// Build it with [`Prompt::text`] and the `with_*` methods. Whether the model can read images
/// or other media depends on the model; `agentai` does not check it, a model without vision
/// support usually answers with an error.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Prompt {
    parts: Vec<Part>,
}

impl Prompt {
    /// Creates an empty prompt.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a prompt with text.
    pub fn text(text: impl Into<String>) -> Self {
        Self::new().with_text(text)
    }

    /// Adds text.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(Part::Text(text.into()));
        self
    }

    /// Adds an image, audio or other file.
    pub fn with_media(mut self, media: Media) -> Self {
        self.parts.push(Part::Media(media));
        self
    }

    /// Adds an image the provider downloads from `url`. See [`Media::image_url`].
    pub fn with_image_url(self, url: impl Into<String>) -> Self {
        self.with_media(Media::image_url(url))
    }

    /// Adds an image from its bytes, for example `image/png`. See [`Media::from_bytes`].
    pub fn with_image_bytes(
        self,
        content_type: impl Into<String>,
        bytes: impl AsRef<[u8]>,
    ) -> Self {
        self.with_media(Media::from_bytes(content_type, bytes))
    }

    /// Returns the parts of the prompt: [`Part::Text`] and [`Part::Media`], in order.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Returns the user message sent to the model.
    pub(crate) fn into_message(self) -> Message {
        Message::new(Role::User, self.parts)
    }
}

/// Anything that can be sent as a [`Prompt`]: text (`&str`, `String` and every other type that
/// converts into `String`), a [`Prompt`] or a single [`Media`].
pub trait IntoPrompt {
    /// Converts the value into a prompt.
    fn into_prompt(self) -> Prompt;
}

impl<T: Into<String>> IntoPrompt for T {
    fn into_prompt(self) -> Prompt {
        Prompt::text(self)
    }
}

impl IntoPrompt for Prompt {
    fn into_prompt(self) -> Prompt {
        self
    }
}

impl IntoPrompt for Media {
    fn into_prompt(self) -> Prompt {
        Prompt::new().with_media(self)
    }
}
