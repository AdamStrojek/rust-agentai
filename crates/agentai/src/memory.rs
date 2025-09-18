use genai::chat::{ChatMessage, ChatRequest, ChatResponse, ChatRole, MessageContent, ToolResponse};

/// This is abstraction of Agent's Memory
pub trait Memory: Send + Sync {
    fn add_message(&mut self, message: ChatMessage);

    fn generate_chat_request(&self) -> ChatRequest;

    fn add_user_message(&mut self, content: MessageContent) {
        self.add_message(ChatMessage::user(content));
    }

    fn add_assistant_message(&mut self, content: MessageContent) {
        self.add_message(ChatMessage::assistant(content));
    }

    fn add_tool_response_message(&mut self, value: ToolResponse) {
        self.add_message(ChatMessage {
            role: ChatRole::Tool, // Tool responses have separate role!
            content: value.into(),
            options: None,
        });
    }

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

/// Memory dedicated to be used in Conversation Mode with Agent
///
/// In most cases this is type of memory that you want to use in your agent. This type of Memory
/// will store whole conversation
///
/// Usage:
/// - multistage process
/// - chat agents
pub struct ConversationMemory {
    chat_request: ChatRequest,
}

impl ConversationMemory {
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
