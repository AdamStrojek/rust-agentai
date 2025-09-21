//! # Agent Memory Management
//!
//! This module provides traits and structs for managing the memory of an AI agent.
//! The `Memory` trait defines a common interface for different memory implementations,
//! allowing developers to customize how an agent stores and retrieves conversation history.
//!
//! ## Available Memory Implementations
//!
//! - `ConversationMemory`: Stores the entire conversation history. This is suitable for
//!   most chat-based agents and multi-stage processes.
use genai::chat::{ChatMessage, ChatRequest, ChatResponse, ChatRole, MessageContent, ToolResponse};

/// An abstraction for an agent's memory.
///
/// The `Memory` trait defines the interface for an agent's memory. It provides methods
/// for adding messages to the history and generating a chat request for the language model.
/// This abstraction allows for different memory strategies to be implemented and used
/// interchangeably with the `Agent`. For example, you could have a memory that stores
/// the full conversation, a summary of it, or only the most recent messages.
pub trait Memory: Send + Sync {
    /// Adds a `ChatMessage` to the memory.
    fn add_message(&mut self, message: ChatMessage);

    /// Generates a `ChatRequest` based on the current state of the memory.
    ///
    /// This method can also be used to signal that the user has finished their turn.
    /// This is a good place to add logic for summarizing or clearing the memory
    /// before the next turn.
    fn generate_chat_request(&self) -> ChatRequest;

    /// A convenience method to add a user message to the memory.
    fn add_user_message(&mut self, content: MessageContent) {
        self.add_message(ChatMessage::user(content));
    }

    /// A convenience method to add an assistant's message to the memory.
    fn add_assistant_message(&mut self, content: MessageContent) {
        self.add_message(ChatMessage::assistant(content));
    }

    /// A convenience method to add a tool response message to the memory.
    fn add_tool_response_message(&mut self, value: ToolResponse) {
        self.add_message(ChatMessage {
            role: ChatRole::Tool, // Tool responses have separate role!
            content: value.into(),
            options: None,
        });
    }

    /// A convenience method to add a `ChatResponse` from the assistant to the memory.
    fn add_response(&mut self, response: &ChatResponse) {
        match response.content.as_ref() {
            Some(content) => {
                self.add_assistant_message(content.clone());
            }
            _ => {
                todo!()
            }
        }
    }
}

/// A `Memory` implementation that stores the entire conversation history.
///
/// This is the most straightforward memory type and is suitable for most use cases,
/// such as chat agents or multi-step tasks where the full context is important.
/// It accumulates messages in a `ChatRequest` and provides the full history
/// for each new request to the language model.
///
/// ## Usage
///
/// This memory is ideal for:
/// - Chatbots that need to remember the entire conversation.
/// - Agents performing multi-stage processes that rely on previous steps.
pub struct ConversationMemory {
    chat_request: ChatRequest,
}

impl ConversationMemory {
    /// Creates a new `ConversationMemory` instance with a system prompt.
    ///
    /// # Arguments
    ///
    /// * `system_prompt` - The initial system message to set the context for the agent.
    pub fn new(system_prompt: &str) -> Self {
        Self {
            chat_request: ChatRequest::from_system(system_prompt),
        }
    }
}

impl Memory for ConversationMemory {
    fn add_message(&mut self, message: ChatMessage) {
        self.chat_request.messages.push(message);
    }

    fn generate_chat_request(&self) -> ChatRequest {
        self.chat_request.clone()
    }
}

// Empty Memory allows you to reuse same `Agent` object each time with clean memory. It doesn't
// store history on previous conversation, only provides system prompt and user prompt to each
// single execution.

// This structure will store partial history only for single iteration inside of agent. It is being
// cleaned when `Agent::run()` returns answer to user.

// This interface is useful when you have same prompt that will be used for multiple sequential calls
// with different set of input data (same system prompt, different user prompt for each execution)
//
// TODO: This will not work when ToolCalls needs previous context. Need to store them
// and find some way to inform memory to clean itself before new Agent execution
// TODO: This would be perfect to automatically cache old prompt and reuse it. Need to identify how
// feasable it is
// pub struct EmptyMemory {
//     system: String,
//     prompt: String,
// }

// impl EmptyMemory {
//     /// Create new empty memory with given system prompt
//     pub fn new(system: &str) -> EmptyMemory {
//         Self {
//             system: system.to_string(), // TODO: Into<String>.into()
//             prompt: String::new(),
//         }
//     }
// }

// impl Memory for EmptyMemory {
//     fn add_message(&mut self, _message: ChatMessage) {}

//     fn add_user_message(&mut self, message: &str) {
//         self.prompt = message.to_string()
//     }

//     fn generate_chat_request(&self) -> ChatRequest {
//         // TODO can generate request when creating new memory, then only clone it
//         ChatRequest::from_system(self.system.to_string())
//             .append_message(ChatMessage::user(self.prompt.clone()))
//     }
// }
