//! The calls of [`SpeechGenerator`]: builders awaited with `.await`.

use crate::generator::SpeechGenerator;
use crate::provider::{AudioError, AudioFormat, SpeechResponse, VoiceId, VoicePreview};
use std::future::{Future, IntoFuture};
use std::pin::Pin;

/// A request to speak a text, made by [`SpeechGenerator::generate`] and sent by awaiting it.
///
/// Unset options are not sent: the provider uses its own defaults.
#[derive(Debug, Clone)]
#[must_use = "a speech request does nothing until it is awaited"]
pub struct SpeechRequest<'a> {
    generator: &'a SpeechGenerator,
    text: String,
    voice: Option<VoiceId>,
    model: Option<String>,
    format: Option<AudioFormat>,
    speed: Option<f64>,
    language: Option<String>,
    instructions: Option<String>,
    timestamps: bool,
    name: Option<String>,
}

impl<'a> SpeechRequest<'a> {
    pub(crate) fn new(generator: &'a SpeechGenerator, text: String) -> Self {
        Self {
            generator,
            text,
            voice: None,
            model: None,
            format: None,
            speed: None,
            language: None,
            instructions: None,
            timestamps: false,
            name: None,
        }
    }

    /// Sets the voice. Without one the API chooses, or fails with
    /// [`AudioError::MissingVoice`] when it needs one (OpenAI, ElevenLabs).
    pub fn with_voice(mut self, voice: impl Into<VoiceId>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Sets the model for this call, instead of the speech model of the provider.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the output format.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the speaking speed.
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Sets the language of the text.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the style instructions. Providers without an equivalent ignore them.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Asks for the time of every word in [`SpeechResponse::timestamps`]. A provider that
    /// cannot give them returns [`AudioError::Unsupported`].
    pub fn with_timestamps(mut self, timestamps: bool) -> Self {
        self.timestamps = timestamps;
        self
    }

    /// Sets the file name of the audio, set as its `Media::name`. Never sent to the provider.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The text to speak.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The voice, if set.
    pub fn voice(&self) -> Option<&str> {
        self.voice.as_deref()
    }

    /// The model of this call, if set.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The output format, if set.
    pub fn format(&self) -> Option<AudioFormat> {
        self.format
    }

    /// The speaking speed, if set.
    pub fn speed(&self) -> Option<f64> {
        self.speed
    }

    /// The language, if set.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// The style instructions, if set.
    pub fn instructions(&self) -> Option<&str> {
        self.instructions.as_deref()
    }

    /// Whether the time of every word is asked for.
    pub fn timestamps(&self) -> bool {
        self.timestamps
    }

    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Fails for a request with timestamps, for providers that cannot give them.
    pub(crate) fn without_timestamps(&self) -> Result<(), AudioError> {
        without_timestamps(self.timestamps)
    }
}

impl<'a> IntoFuture for SpeechRequest<'a> {
    type Output = Result<SpeechResponse, AudioError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.generator.speak(&self).await })
    }
}

/// One turn of a dialogue: who speaks and what.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DialogueTurn {
    /// The voice that speaks the turn.
    pub voice: VoiceId,
    /// What the voice says.
    pub text: String,
}

impl DialogueTurn {
    /// Creates a turn of `voice` saying `text`.
    pub fn new(voice: impl Into<VoiceId>, text: impl Into<String>) -> Self {
        Self {
            voice: voice.into(),
            text: text.into(),
        }
    }
}

/// A request to speak a dialogue of several voices, made by
/// [`SpeechGenerator::generate_dialogue`] and sent by awaiting it.
#[derive(Debug, Clone)]
#[must_use = "a dialogue request does nothing until it is awaited"]
pub struct DialogueRequest<'a> {
    generator: &'a SpeechGenerator,
    voices: Vec<VoiceId>,
    turns: Vec<(usize, String)>,
    model: Option<String>,
    format: Option<AudioFormat>,
    speed: Option<f64>,
    language: Option<String>,
    timestamps: bool,
    name: Option<String>,
}

impl<'a> DialogueRequest<'a> {
    pub(crate) fn new(
        generator: &'a SpeechGenerator,
        dialogue: impl IntoIterator<Item = DialogueTurn>,
    ) -> Self {
        let mut voices: Vec<VoiceId> = Vec::new();
        let turns = dialogue
            .into_iter()
            .map(|turn| {
                let speaker = match voices.iter().position(|voice| *voice == turn.voice) {
                    Some(speaker) => speaker,
                    None => {
                        voices.push(turn.voice);
                        voices.len() - 1
                    }
                };
                (speaker, turn.text)
            })
            .collect();
        Self {
            generator,
            voices,
            turns,
            model: None,
            format: None,
            speed: None,
            language: None,
            timestamps: false,
            name: None,
        }
    }

    /// Sets the model for this call, instead of the dialogue model of the provider.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the output format.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the speaking speed of every turn. A provider that cannot set it for a dialogue
    /// returns [`AudioError::Unsupported`].
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    /// Sets the language of the text.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Asks for the time of every word in [`SpeechResponse::timestamps`]. A provider that
    /// cannot give them returns [`AudioError::Unsupported`].
    pub fn with_timestamps(mut self, timestamps: bool) -> Self {
        self.timestamps = timestamps;
        self
    }

    /// Sets the file name of the audio, set as its `Media::name`. Never sent to the provider.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The distinct voices, in the order they first speak. A voice's index is its speaker
    /// number in [`turns`](Self::turns).
    pub fn voices(&self) -> &[VoiceId] {
        &self.voices
    }

    /// Every turn as the index of its voice in [`voices`](Self::voices) and its text.
    pub fn turns(&self) -> &[(usize, String)] {
        &self.turns
    }

    /// The speaking speed, if set.
    pub fn speed(&self) -> Option<f64> {
        self.speed
    }

    /// The model of this call, if set.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The output format, if set.
    pub fn format(&self) -> Option<AudioFormat> {
        self.format
    }

    /// The language, if set.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// Whether the time of every word is asked for.
    pub fn timestamps(&self) -> bool {
        self.timestamps
    }

    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Fails for a request with timestamps, for providers that cannot give them.
    pub(crate) fn without_timestamps(&self) -> Result<(), AudioError> {
        without_timestamps(self.timestamps)
    }
}

impl<'a> IntoFuture for DialogueRequest<'a> {
    type Output = Result<SpeechResponse, AudioError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.generator.speak_dialogue(&self).await })
    }
}

/// A request to design voices from a description, made by [`SpeechGenerator::design_voice`]
/// and sent by awaiting it. It returns previews to listen to.
#[derive(Debug, Clone)]
#[must_use = "a voice design request does nothing until it is awaited"]
pub struct DesignVoiceRequest<'a> {
    generator: &'a SpeechGenerator,
    description: String,
    text: Option<String>,
    language: Option<String>,
    count: usize,
    model: Option<String>,
}

impl<'a> DesignVoiceRequest<'a> {
    pub(crate) fn new(generator: &'a SpeechGenerator, description: String) -> Self {
        Self {
            generator,
            description,
            text: None,
            language: None,
            count: 1,
            model: None,
        }
    }

    /// Sets the text the previews say, for providers that let you choose.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Sets the language of the voice (a BCP-47 code such as `pl`). Fish Audio and oMLX send it;
    /// ElevenLabs has no such setting and ignores it.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets how many previews to ask for (1 by default). A provider may return its own number.
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }

    /// Sets the model for this call, instead of the design model of the provider. It does not
    /// turn on voice design for a provider without a design model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The description of the voice.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The text of the previews, if set.
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// The language of the voice, if set.
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// How many previews to ask for.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The model of this call, if set.
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

impl<'a> IntoFuture for DesignVoiceRequest<'a> {
    type Output = Result<Vec<VoicePreview>, AudioError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move { self.generator.design(&self).await })
    }
}

fn without_timestamps(timestamps: bool) -> Result<(), AudioError> {
    if timestamps {
        Err(AudioError::Unsupported("word timestamps"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::SpeechProvider;
    use async_trait::async_trait;

    struct NoProvider;

    #[async_trait]
    impl SpeechProvider for NoProvider {
        async fn generate(&self, _: &SpeechRequest<'_>) -> Result<SpeechResponse, AudioError> {
            unreachable!()
        }
    }

    #[test]
    fn dialogue_voices_are_distinct_in_order_of_first_turn() {
        let generator = SpeechGenerator::from_provider(NoProvider);
        let request = generator.generate_dialogue([
            DialogueTurn::new("b", "1"),
            DialogueTurn::new("a", "2"),
            DialogueTurn::new("b", "3"),
        ]);
        assert_eq!(request.voices(), ["b", "a"]);
        assert_eq!(
            request.turns(),
            [
                (0, "1".to_string()),
                (1, "2".to_string()),
                (0, "3".to_string())
            ]
        );
    }
}
