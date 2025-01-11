use crate::AgentTool;
use anyhow::{Context, Result};
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, JsonSpec, Tool, ToolResponse};
use genai::Client;
use log::{debug, trace};
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde_json::{from_str, json, Value};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

/// The `Agent` struct represents an agent that interacts with a chat model.
/// It maintains a history of chat messages, a set of tools, and a context.
///
/// As `Context` you can provide any structure. Such object will not be used by
/// `Agent` itself, but it will be passed in unmodified state as reference to any
/// `AgentTool` trait, that was registered to be used.
pub struct Agent<'a, CTX> {
    /// Reference to GenAI Client
    client: &'a Client,
    /// Dynamic Context
    context: &'a CTX,
    /// Objects of tools implementations
    tools_impl: HashMap<String, Arc<dyn AgentTool<CTX>>>,
    tools_defs: Vec<Tool>,
    history: Vec<ChatMessage>,
}

impl<'a, CTX> Agent<'a, CTX> {
    /// Creates a new `Agent` instance.
    ///
    /// # Arguments
    ///
    /// * `client` - The client used to interact with the chat model.
    /// * `system` - The system message to initialize the chat history.
    /// * `context` - The context associated with the agent.
    ///
    /// # Returns
    ///
    /// A new `Agent` instance.
    pub fn new(client: &'a Client, system: &str, context: &'a CTX) -> Self {
        Self {
            client,
            context,
            tools_impl: HashMap::new(),
            tools_defs: vec![],
            history: vec![ChatMessage::system(system.trim())],
        }
    }

    /// Adds a tool to the agent.
    ///
    /// # Arguments
    ///
    /// * `agent_tool` - The tool to add.
    pub fn add_tool(&mut self, agent_tool: Arc<dyn AgentTool<CTX>>) {
        let tool = Tool::new(agent_tool.name())
            .with_description(agent_tool.description())
            .with_schema(agent_tool.schema());
        self.tools_defs.push(tool);

        self.tools_impl.insert(agent_tool.name(), agent_tool);
    }

    /// Runs the agent with the given model and prompt.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use for the chat.
    /// * `prompt` - The prompt to send to the chat model.
    ///
    /// # Returns
    ///
    /// A result containing the deserialized response.
    pub async fn run<D>(&mut self, model: &str, prompt: &str) -> Result<D>
    where
        D: DeserializeOwned + JsonSchema + 'static,
    {
        debug!("Agent Question: {}", prompt);
        // Add new request to history
        // TODO: What to do when message have images? Should we send them only once?
        self.history.push(ChatMessage::user(prompt));

        // Prepare chat options
        // TODO: Allow to provide chat options
        let mut chat_opts = ChatOptions::default().with_temperature(0.2);

        let is_answer_string = TypeId::of::<String>() == TypeId::of::<D>();
        if !is_answer_string {
            // If answer type is more complex then add response format to request options
            let mut response_schema = serde_json::to_value(schema_for!(D))?;
            let obj = response_schema.as_object_mut().unwrap();
            // Schemars attaches additional fields and not every LLM accepts them (Gemini)
            obj.remove("$schema");
            obj.remove("title");
            chat_opts = chat_opts.with_response_format(JsonSpec::new("ResponseFormat", json!(obj)));
        }

        loop {
            // Create chat request
            let mut chat_req = ChatRequest::new(self.history.clone());
            if !self.tools_defs.is_empty() {
                chat_req = chat_req.with_tools(self.tools_defs.clone());
            }
            let chat_resp = self
                .client
                .exec_chat(model, chat_req, Some(&chat_opts))
                .await?;
            let chat_resp_str = chat_resp.content_text_as_str();

            if let Some(tools_call) = chat_resp.clone().into_tool_calls() {
                self.history.push(ChatMessage::from(tools_call.clone()));
                // Go through tool use
                for tool_request in tools_call {
                    trace!("Tool request: {}", tool_request.fn_name);
                    if let Some(tool) = self.tools_impl.get(&tool_request.fn_name) {
                        let result = tool
                            .call(
                                self.context,
                                serde_json::from_value(tool_request.fn_arguments)?,
                            )
                            .await?;
                        trace!("Tool result: {}", result);
                        self.history.push(ChatMessage::from(ToolResponse::new(
                            tool_request.call_id.clone(),
                            result,
                        )));
                    } else {
                        trace!("No tool found for {}", tool_request.fn_name);
                    }
                }
            } else {
                let mut resp = chat_resp_str
                    .context("Missing string in response")?
                    .to_string();
                debug!("Agent Answer: {resp}");
                self.history.push(ChatMessage::assistant(resp.clone()));
                if is_answer_string {
                    // TODO: Workaround when choosing String as response type. Because we are
                    // expecting D: DeserializeOwned then we can't return String directly.
                    // To workaround this I escape content and later deserialize it using
                    // serde_json::from_str to correct "struct" (String)
                    resp = Value::String(resp).to_string();
                }
                let resp = from_str(&resp)?;
                return Ok(resp);
            }
        }
    }
}
