//! # Processing the history before it is sent
//!
//! Every model call sends the whole conversation again. A [`HistoryProcessor`] gets the
//! messages right before each call and returns the ones to send, so it can drop, shorten or
//! summarize them. Only the request changes: the history you own and the messages of the run
//! keep everything.
//!
//! [`HistoryPruning`] is a ready-made processor: it removes media and the model's reasoning from
//! the turns that are finished. A closure `Fn(Vec<Message>) -> Vec<Message>` is a processor too.
//!
//! Add processors with `Agent::with_history_processor` in `agentai`.

use crate::message::{Media, Message, Part, Role};
use async_trait::async_trait;

/// Error returned by a [`HistoryProcessor`]. It stops the run with `AgentError::History` in
/// `agentai`.
pub type HistoryError = Box<dyn std::error::Error + Send + Sync>;

/// Changes the messages sent to the model.
///
/// Called before every model call of a run with the history followed by the messages of the
/// run so far, including the tool results of an unfinished tool loop. Keep an assistant
/// message with tool calls together with its tool results, and keep the reasoning of an
/// unfinished tool loop: Anthropic and Gemini reject the request otherwise.
///
/// The processor runs before every call, so return the same messages for the same input:
/// prompt caching only works when what was sent before is sent the same way again.
///
/// ```rust
/// use agentai_core::history::{HistoryError, HistoryProcessor};
/// use agentai_core::message::Message;
/// use async_trait::async_trait;
///
/// /// Sends only the last messages.
/// struct KeepLast(usize);
///
/// #[async_trait]
/// impl HistoryProcessor for KeepLast {
///     async fn process(&self, messages: Vec<Message>) -> Result<Vec<Message>, HistoryError> {
///         // A real processor must not separate tool calls from their results
///         let start = messages.len().saturating_sub(self.0);
///         Ok(messages[start..].to_vec())
///     }
/// }
/// ```
#[async_trait]
pub trait HistoryProcessor: Send + Sync {
    /// Returns the messages to send instead of `messages`.
    async fn process(&self, messages: Vec<Message>) -> Result<Vec<Message>, HistoryError>;
}

#[async_trait]
impl<F> HistoryProcessor for F
where
    F: Fn(Vec<Message>) -> Vec<Message> + Send + Sync,
{
    async fn process(&self, messages: Vec<Message>) -> Result<Vec<Message>, HistoryError> {
        Ok(self(messages))
    }
}

/// A [`HistoryProcessor`] that removes media and reasoning from the finished turns. Nothing is
/// removed by default.
///
/// A turn is finished once the model gave its final answer: an assistant message without tool
/// calls. Messages after the last such answer, for example the tool loop of the current run,
/// are kept as they are.
///
/// ```rust
/// # use agentai_core::history::HistoryPruning;
/// # use agentai_core::message::Message;
/// # let history: Vec<Message> = Vec::new();
/// let pruning = HistoryPruning::default().with_media(true).with_reasoning(true);
/// // Usually passed to `Agent::with_history_processor`; it can also be called directly
/// let pruned: Vec<Message> = pruning.prune(&history);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct HistoryPruning {
    /// Replace media (images, audio, files) in user messages and tool results with a short
    /// text such as `[omitted image/png: cat.png]`, so the model knows a file was there. Tools
    /// that return files should also say how to get them again, for example by an ID.
    ///
    /// The reasoning that follows replaced media in the same turn is removed too, even
    /// without [`reasoning`](HistoryPruning::reasoning): Anthropic rejects thinking blocks
    /// whose earlier content changed.
    pub media: bool,
    /// Remove the model's reasoning: [`Part::Reasoning`], [`Part::ReasoningSignature`],
    /// [`Part::ProviderData`] and the signatures of tool calls.
    pub reasoning: bool,
}

impl HistoryPruning {
    /// Sets whether media in finished turns is replaced with text.
    pub fn with_media(mut self, media: bool) -> Self {
        self.media = media;
        self
    }

    /// Sets whether reasoning in finished turns is removed.
    pub fn with_reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Returns the history with the finished turns pruned. Messages left without any part are
    /// dropped.
    pub fn prune(&self, history: &[Message]) -> Vec<Message> {
        let finished = history
            .iter()
            .rposition(|message| message.role == Role::Assistant && message.tool_calls().is_empty())
            .map_or(0, |last_answer| last_answer + 1);
        let (old, current) = history.split_at(finished);
        // Set once media of the current turn was replaced: the reasoning after it is stale
        let mut media_replaced = false;
        let mut pruned = Vec::with_capacity(history.len());
        for message in old {
            if message.role == Role::User {
                media_replaced = false;
            }
            let mut message = message.clone();
            if self.reasoning || media_replaced {
                message = message.without_reasoning();
            }
            if self.media && has_media(&message.parts) {
                message = message.without_media();
                media_replaced = true;
            }
            if !message.parts.is_empty() {
                pruned.push(message);
            }
        }
        pruned.extend(current.iter().cloned());
        pruned
    }
}

#[async_trait]
impl HistoryProcessor for HistoryPruning {
    async fn process(&self, messages: Vec<Message>) -> Result<Vec<Message>, HistoryError> {
        Ok(self.prune(&messages))
    }
}

impl Message {
    /// Returns the message with every media part, also inside tool results, replaced with a
    /// short text such as `[omitted image/png: cat.png]`.
    pub fn without_media(mut self) -> Self {
        self.parts = parts_without_media(self.parts);
        self
    }

    /// Returns the message without the model's reasoning: [`Part::Reasoning`],
    /// [`Part::ReasoningSignature`], [`Part::ProviderData`] and the signatures of tool calls.
    ///
    /// Only remove it from finished turns: during a tool loop, Anthropic and Gemini need the
    /// reasoning back unchanged.
    ///
    /// Anthropic also rejects a stored history whose thinking was created with another system
    /// prompt or other tools. After changing them, remove the reasoning once and save the
    /// history again:
    ///
    /// ```rust
    /// # use agentai_core::message::Message;
    /// # let history: Vec<Message> = Vec::new();
    /// let history: Vec<Message> = history
    ///     .into_iter()
    ///     .map(Message::without_reasoning)
    ///     .filter(|message| !message.parts.is_empty())
    ///     .collect();
    /// ```
    pub fn without_reasoning(mut self) -> Self {
        self.parts.retain(|part| {
            !matches!(
                part,
                Part::Reasoning(_) | Part::ReasoningSignature(_) | Part::ProviderData(_)
            )
        });
        for part in &mut self.parts {
            if let Part::ToolCall(call) = part {
                call.signatures.clear();
            }
        }
        self
    }
}

fn has_media(parts: &[Part]) -> bool {
    parts.iter().any(|part| match part {
        Part::Media(_) => true,
        Part::ToolResponse(response) => has_media(&response.content),
        _ => false,
    })
}

fn parts_without_media(parts: Vec<Part>) -> Vec<Part> {
    parts
        .into_iter()
        .map(|part| match part {
            Part::Media(media) => Part::Text(media_placeholder(&media)),
            Part::ToolResponse(mut response) => {
                response.content = parts_without_media(response.content);
                Part::ToolResponse(response)
            }
            part => part,
        })
        .collect()
}

fn media_placeholder(media: &Media) -> String {
    match &media.name {
        Some(name) => format!("[omitted {}: {name}]", media.content_type),
        None => format!("[omitted {}]", media.content_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ToolCall, ToolOutput, ToolResponse};
    use serde_json::json;

    fn image() -> Media {
        Media::from_base64("image/png", "aGk=").with_name("cat.png")
    }

    /// One finished turn with a tool round, then a tool loop that has not answered yet.
    fn history() -> Vec<Message> {
        let call = ToolCall::new("1", "draw", json!({})).with_signatures(vec!["sig".into()]);
        let tool_use = Message::new(
            Role::Assistant,
            vec![
                Part::Reasoning("let me draw".into()),
                Part::ReasoningSignature("sig".into()),
                Part::ToolCall(call.clone()),
            ],
        );
        let result = Message::tool_response(ToolResponse::new(
            &call,
            ToolOutput::from_text("Drew it").with_media(image()),
        ));
        vec![
            Message::new(
                Role::User,
                vec![Part::Text("draw this".into()), Part::Media(image())],
            ),
            tool_use.clone(),
            result.clone(),
            // Only reasoning: dropped once the reasoning is removed
            Message::new(Role::Assistant, vec![Part::Reasoning("done".into())]),
            Message::assistant("Here it is"),
            Message::user("again"),
            tool_use,
            result,
        ]
    }

    #[test]
    fn prunes_finished_turns_and_keeps_the_unfinished_one() {
        let history = history();
        let pruned = HistoryPruning::default()
            .with_media(true)
            .with_reasoning(true)
            .prune(&history);

        let call = ToolCall::new("1", "draw", json!({}));
        let expected_old = [
            Message::new(
                Role::User,
                vec![
                    Part::Text("draw this".into()),
                    Part::Text("[omitted image/png: cat.png]".into()),
                ],
            ),
            Message::new(Role::Assistant, vec![Part::ToolCall(call.clone())]),
            Message::tool_response(ToolResponse::new(
                &call,
                ToolOutput::from_text("Drew it").with_text("[omitted image/png: cat.png]"),
            )),
            Message::assistant("Here it is"),
        ];
        assert_eq!(pruned[..4], expected_old[..]);
        assert_eq!(pruned[4..], history[5..]);
    }

    #[test]
    fn media_alone_removes_the_reasoning_after_the_replaced_media_in_its_turn() {
        let history = history();
        let pruned = HistoryPruning::default().with_media(true).prune(&history);

        let bare_call = ToolCall::new("1", "draw", json!({}));
        // The user's image is replaced, so the tool call after it loses its reasoning
        assert_eq!(
            pruned[1],
            Message::new(Role::Assistant, vec![Part::ToolCall(bare_call)])
        );
        assert_eq!(pruned[4..], history[5..]);

        // A turn without media keeps its reasoning
        let call = ToolCall::new("1", "draw", json!({}));
        let text_only = vec![
            Message::user("hi"),
            history[1].clone(),
            Message::tool_response(ToolResponse::new(&call, "Drew it")),
            Message::assistant("done"),
        ];
        assert_eq!(
            HistoryPruning::default().with_media(true).prune(&text_only),
            text_only
        );
    }

    #[test]
    fn a_history_without_a_final_answer_is_kept() {
        let history = history()[..3].to_vec();
        let pruned = HistoryPruning::default()
            .with_media(true)
            .with_reasoning(true)
            .prune(&history);
        assert_eq!(pruned, history);
    }
}
