//! # Core components of AI Agents
//!
//! This module contains core components that you can use in your AI Agents
//!
//! To read more about Agents look into [crate::agent::Agent]
//!
//! To read more about structured output look into [crate::structured_output]
//!
//! To read more about tool look into [crate::tool]

use crate::tool::ToolBox;
use anyhow::{anyhow, Result};
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, JsonSpec, MessageContent, ToolResponse};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ClientBuilder, ModelIden, ServiceTarget};
use log::{debug, trace};
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde_json::{from_str, json, Value};
use std::any::TypeId;
use std::sync::Arc;

/// The `Agent` struct represents an agent that interacts with a chat model.
/// It maintains a history of chat messages, a set of tools, and a context.
///
/// As `Context` you can provide any structure. Such object will not be used by
/// `Agent` itself, but it will be passed in unmodified state as reference to any
/// `AgentTool` trait, that was registered to be used.
#[derive(Clone)]
pub struct Agent {
    /// Reference to GenAI Client
    client: Client,

    // tool_box: impl ToolBox,
    history: Vec<ChatMessage>,
}

impl Agent {
    /// Creates a new `Agent` instance.
    ///
    /// This creation method will create a new agent instance with a default GenAI client
    ///
    /// # Arguments
    ///
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Returns
    ///
    /// A new `Agent` instance.
    pub fn new(system: &str) -> Self {
        let client = Client::default();

        Self::new_with_client(client, system)
    }

    /// Creates a new `Agent` instance with provided GenAI Client
    ///
    /// # Arguments
    ///
    /// * `client` - User provided GenAI Client
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Returns
    ///
    /// A new `Agent` instance.
    pub fn new_with_client(client: Client, system: &str) -> Self {
        Self {
            client,
            history: vec![ChatMessage::system(system.trim())],
        }
    }

    pub fn new_with_url(base_url: &str, api_key: &str, system: &str) -> Self {
        let endpoint = Endpoint::from_owned(Arc::from(base_url));
        let auth = AuthData::from_single(api_key);
       	let target_resolver = ServiceTargetResolver::from_resolver_fn(
            |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
                let ServiceTarget { model, .. } = service_target;
                let model = ModelIden::new(AdapterKind::OpenAI, model.model_name);
                Ok(ServiceTarget { endpoint, auth, model })
            },
        );
        let client = ClientBuilder::default()
            .with_service_target_resolver(target_resolver).build();
        Self::new_with_client(client, system)
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
    ///
    /// ## Structured Output
    /// Type returned by this function is responsible for setting LLM response into structured output
    ///
    /// For more information go to [crate::structured_output]
    pub async fn run<D>(&mut self, model: &str, prompt: &str, toolbox: Option<&dyn ToolBox>) -> Result<D>
    where
        D: DeserializeOwned + JsonSchema + 'static,
    {
        // TODO change returned type
        // Need to create new type that will provide not only response structure,
        // but also statistics and reasoning.
        debug!("Agent Question: {}", prompt);
        // Add new request to history
        // TODO: Create new history trait
        // This will allow on configuring behaviour of messages. When doing multi-agent
        // approach we could decide what history is being used, should we save all messages etc.
        // TODO: What to do when message have images? Should we send them only once?
        self.history.push(ChatMessage::user(prompt));

        // Prepare chat options
        // TODO: Allow to provide chat options to GenAI
        // This should be be part
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

        // TODO move it to config structure
        let max_iterations = 5;

        for iteration in 0..max_iterations {
            debug!("Agent iteration: {}", iteration);
            // Create chat request
            let mut chat_req = ChatRequest::new(self.history.clone());
            if let Some(ref toolbox) = toolbox {
                chat_req = chat_req.with_tools(toolbox.tools_definitions()?);
            }
            let chat_resp = self
                .client
                .exec_chat(model, chat_req, Some(&chat_opts))
                .await?;

            match chat_resp.content {
                Some(MessageContent::Text(text)) => {
                    let mut resp = text;
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
                },
                Some(MessageContent::ToolCalls(tools_call)) => {
                    self.history.push(ChatMessage::from(tools_call.clone()));
                    // Go through tool use
                    for tool_request in tools_call {
                        trace!("Tool request: {} with arguments: {}", tool_request.fn_name, tool_request.fn_arguments.to_string());
                        if let Some(ref tool) = toolbox {
                            match tool.call_tool(tool_request.fn_name, tool_request.fn_arguments).await {
                                Ok(result) => {
                                    trace!("Tool result: {}", result);
                                    self.history.push(ChatMessage::from(ToolResponse::new(
                                        tool_request.call_id.clone(),
                                        result,
                                    )));
                                },
                                Err(err) => {
                                    // If MCP Server fails we need to redirect this information to model
                                    // this will allow to react on what happens. Some MCP Servers returns
                                    // important information as error for Agent
                                    // TODO: Allow user to configure this behaviour. Depending on MCP
                                    // server this may contain important information, or this may be
                                    // indication of unrecoverable failure
                                    trace!("Error: {}", err);
                                    self.history.push(ChatMessage::from(ToolResponse::new(
                                        tool_request.call_id.clone(),
                                        err.to_string(),
                                    )));
                                }
                            };
                        } else {
                            todo!("No tool found for {}", tool_request.fn_name);
                        }
                    }
                },
                Some(msg_content) => {
                    return Err(anyhow!(format!("Unsupported message content {:?}", msg_content)));
                },
                None => {}
            };
        }

        Err(anyhow!(format!("Unable to get response in {max_iterations} tries")))
    }
}
