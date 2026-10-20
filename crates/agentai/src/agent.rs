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
use anyhow::{Result, anyhow};
use genai::Client;
use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, JsonSpec, ToolCall, ToolResponse};
use genai::resolver::{AuthData, Endpoint};
use log::{debug, trace};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde_json::{Value, from_str, json};
use std::any::TypeId;

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
    /// This creation method will create a new agent instance with a default GenAI client.
    /// genai picks the provider from the model name and reads the provider's API key
    /// from its usual environment variable.
    ///
    /// # Arguments
    ///
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Errors
    ///
    /// Returns an error when the GenAI client cannot be created.
    pub fn new(system: &str) -> Result<Self> {
        let client = Client::new()?;

        Ok(Self::new_with_client(client, system))
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

    /// Creates a new `Agent` instance that talks to an OpenAI-compatible endpoint
    /// (OpenAI, OpenRouter, LiteLLM, Ollama, ...).
    ///
    /// Every model name is sent through genai's OpenAI adapter to `base_url`, authenticated
    /// with `api_key`. Model names are passed as they are, so names like `openai/gpt-4.1-mini`
    /// or `vendor/model:free` work. A name with another genai namespace, such as
    /// `anthropic::claude-sonnet-4-5`, fails with an adapter mismatch error.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the API, for example `https://openrouter.ai/api/v1/`.
    /// * `api_key` - API key sent with every request.
    /// * `system` - The system message to initialize the chat history.
    ///
    /// # Errors
    ///
    /// Returns an error when the GenAI client cannot be created.
    pub fn new_with_url(base_url: &str, api_key: &str, system: &str) -> Result<Self> {
        let client = Client::builder()
            .with_adapter_kind(AdapterKind::OpenAI)
            .append_provider_config(
                AdapterKind::OpenAI,
                (
                    Endpoint::from_owned(base_url.to_string()),
                    AuthData::from_single(api_key),
                ),
            )
            .build()?;
        Ok(Self::new_with_client(client, system))
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
            debug!("Agent iteration: {iteration}");
            // Create chat request
            let mut chat_req = ChatRequest::new(self.history.clone());
            if let Some(toolbox) = toolbox {
                chat_req = chat_req.with_tools(toolbox.tools_definitions()?);
            }
            let chat_resp = self
                .client
                .exec_chat(model, chat_req, Some(&chat_opts))
                .await?;

            // Collect tool calls before the response is consumed. Some models return text
            // (e.g. "Let me check...") together with tool calls; the tool calls win.
            let tool_calls: Vec<ToolCall> = chat_resp.tool_calls().into_iter().cloned().collect();

            if !tool_calls.is_empty() {
                // Keeps text, reasoning and provider continuation data next to the tool calls
                if let Some(assistant_msg) = chat_resp.into_assistant_message_for_tool_use() {
                    self.history.push(assistant_msg);
                }
                for tool_call in tool_calls {
                    trace!(
                        "Tool request: {} with arguments: {}",
                        tool_call.fn_name, tool_call.fn_arguments
                    );
                    let content = match toolbox {
                        Some(toolbox) => {
                            match toolbox
                                .call_tool(
                                    tool_call.fn_name.clone(),
                                    tool_call.fn_arguments.clone(),
                                )
                                .await
                            {
                                Ok(result) => {
                                    trace!("Tool result: {result}");
                                    result
                                }
                                Err(err) => {
                                    // Tool errors go back to the model, so it can react to them.
                                    // Some MCP servers return important information as errors.
                                    // TODO: Allow user to configure this behaviour. Depending on
                                    // the tool this may also mean an unrecoverable failure
                                    trace!("Tool error: {err}");
                                    err.to_string()
                                }
                            }
                        }
                        None => {
                            debug!(
                                "Model requested tool '{}' but no toolbox was provided",
                                tool_call.fn_name
                            );
                            format!(
                                "Tool '{}' is not available: no tools were provided",
                                tool_call.fn_name
                            )
                        }
                    };
                    self.history
                        .push(ToolResponse::from_tool_call(&tool_call, content).into());
                }
                continue;
            }

            let texts = chat_resp.content.into_texts();
            if texts.is_empty() {
                // Nothing usable in this response, ask again
                continue;
            }
            // Structured output may be split across several text parts
            let mut resp = texts.join("");
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

        Err(anyhow!(format!(
            "Unable to get response in {max_iterations} tries"
        )))
    }
}
