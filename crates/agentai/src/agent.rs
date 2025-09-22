//! # Core components of AI Agents
//!
//! This module contains core components that you can use in your AI Agents
//!
//! To read more about Agents look into [crate::agent::Agent]
//!
//! To read more about structured output look into [crate::structured_output]
//!
//! To read more about tool look into [crate::tool]

use crate::memory::{ConversationMemory, Memory};
use crate::tool::ToolBox;
use anyhow::{anyhow, Result};
use genai::adapter::AdapterKind;
use genai::chat::{ChatOptions, JsonSpec, MessageContent, ToolResponse};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client as GenAIClient, ClientBuilder, ModelIden, ServiceTarget};
use log::{debug, trace};
use schemars::{schema_for, JsonSchema};
use serde::de::DeserializeOwned;
use serde_json::{from_str, json, Value};
use std::any::TypeId;
use std::sync::Arc;
use type_state_builder::TypeStateBuilder;

/// The `Agent` struct represents an agent that interacts with a chat model.
/// It maintains a history of chat messages, and a set of tools.
///
/// You can construct a new `Agent` instance using the `AgentBuilder`, which is accessible
/// via the `builder()` method.
// #[derive(Clone)]
#[derive(TypeStateBuilder)]
#[builder(setter_prefix = "with_")]
pub struct Agent {
    /// GenAI Client
    client: GenAIClient,

    #[builder(required)]
    memory: Box<dyn Memory>,
}

fn genai_client_with_url(base_url: &str, api_key: &str) -> GenAIClient {
    let endpoint = Endpoint::from_owned(Arc::from(base_url));
    let auth = AuthData::from_single(api_key);
    let target_resolver = ServiceTargetResolver::from_resolver_fn(
        |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
            let ServiceTarget { model, .. } = service_target;
            let model = ModelIden::new(AdapterKind::OpenAI, model.model_name);
            Ok(ServiceTarget {
                endpoint,
                auth,
                model,
            })
        },
    );
    ClientBuilder::default()
        .with_service_target_resolver(target_resolver)
        .build()
}

impl AgentBuilder_MissingMemory {
    /// Sets system prompt for agent using `ConversationMemory` struct
    ///
    /// You need to choose do you want to initialize your memory using `with_memory`
    /// or `with_system_prompt` function
    pub fn with_system_prompt(self, system_prompt: &str) -> AgentBuilder_HasMemory {
        self.with_memory(Box::new(ConversationMemory::new(system_prompt)))
    }

    pub fn with_url(self, base_url: &str, api_key: &str) -> Self {
        self.with_client(genai_client_with_url(base_url, api_key))
    }
}

impl AgentBuilder_HasMemory {
    pub fn with_url(self, base_url: &str, api_key: &str) -> Self {
        self.with_client(genai_client_with_url(base_url, api_key))
    }
}

impl Agent {
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
    pub async fn run<D>(
        &mut self,
        model: &str,
        prompt: &str,
        toolbox: Option<&dyn ToolBox>,
    ) -> Result<D>
    where
        D: DeserializeOwned + JsonSchema + 'static,
    {
        // TODO change returned type
        // Need to create new type that will provide not only response structure,
        // but also statistics and reasoning.
        debug!("Agent Question: {prompt}");
        // Add new request to history
        // TODO: Create new history trait
        // This will allow on configuring behaviour of messages. When doing multi-agent
        // approach we could decide what history is being used, should we save all messages etc.
        // TODO: What to do when message have images? Should we send them only once?
        self.memory.add_user_message(prompt.into());

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
            debug!("Agent iteration: {iteration}");
            // Create chat request
            let mut chat_req = self.memory.generate_chat_request();
            // TODO: Should this be moved to Memory trait?
            if let Some(toolbox) = toolbox {
                chat_req = chat_req.with_tools(toolbox.tools_definitions()?);
            }

            // ---------------
            // Sending request to LLM
            let chat_resp = self
                .client
                .exec_chat(model, chat_req, Some(&chat_opts))
                .await?;

            debug!("Chat Resp: {chat_resp:#?}");

            // ---------------
            // Saving LLM response in memory
            self.memory.add_response(&chat_resp);

            // ---------------
            // Should I return message or perform additional calls?
            match chat_resp.content {
                Some(MessageContent::Text(text)) => {
                    let mut resp = text;
                    debug!("Agent Answer: {resp}");
                    if is_answer_string {
                        // TODO: Workaround when choosing String as response type. Because we are
                        // expecting D: DeserializeOwned then we can't return String directly.
                        // To workaround this I escape content and later deserialize it using
                        // serde_json::from_str to correct "struct" (String)
                        resp = Value::String(resp).to_string();
                    }

                    // Performing deserialization, LLM always deliver message as stringify JSON
                    let resp = from_str(&resp)?;
                    return Ok(resp);
                }
                Some(MessageContent::ToolCalls(tools_call)) => {
                    // Go through tool use
                    for tool_request in tools_call {
                        trace!(
                            "Tool request: {} with arguments: {}",
                            tool_request.fn_name,
                            tool_request.fn_arguments
                        );
                        if let Some(tool) = toolbox {
                            let tool_response_content = match tool
                                .call_tool(tool_request.fn_name, tool_request.fn_arguments)
                                .await
                            {
                                Ok(content) => content,
                                Err(err) => err.to_string(),
                            };
                            trace!("Tool result: {tool_response_content}");
                            let tool_response = ToolResponse::new(
                                tool_request.call_id.clone(),
                                tool_response_content,
                            );
                            self.memory.add_tool_response_message(tool_response);
                        } else {
                            todo!("No tool found for {}", tool_request.fn_name);
                        }
                    }
                }
                Some(msg_content) => {
                    return Err(anyhow!(format!(
                        "Unsupported message content {:?}",
                        msg_content
                    )));
                }
                None => {}
            };
        }

        Err(anyhow!(format!(
            "Unable to get response in {max_iterations} tries"
        )))
    }
}
