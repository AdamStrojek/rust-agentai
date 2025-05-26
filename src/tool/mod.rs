pub mod websearch;

#[cfg(feature = "mcp-client")]
pub mod mcp;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

/// Represents a tool that can be used by an agent (e.g., an LLM).
/// Each tool provides a specific capability with a defined name, description,
/// schema for its parameters, and an asynchronous call method.
#[async_trait]
pub trait Tool {
    /// Returns the name of the tool. This name is used by the agent to refer to the tool.
    fn name(&self) -> String;

    /// Returns a description of the tool. This description is used by the agent
    /// to understand what the tool does and when to use it.
    fn description(&self) -> String;

    /// Returns the JSON schema defining the parameters required by the tool's `call` method.
    fn schema(&self) -> Value;
    // TODO: Maybe do dynamic parameters type?
    // type Params: DeserializeOwned + JsonSchema;
    // {
    // 	let mut schema = serde_json::to_value(schema_for!(Self::Params)).unwrap();
    // 	let mut obj = schema.as_object_mut().unwrap();
    // 	obj.remove("$schema");
    // 	obj.remove("title");
    // 	json!(obj)
    // }

    /// Executes the tool with the given parameters.
    /// The parameters are expected to conform to the schema returned by `schema()`.
    async fn call(&self, params: Value) -> anyhow::Result<String>;
}

/// A container or manager for a collection of `Tool` instances.
/// This provides a way to group and access multiple tools.
pub trait ToolBox {
    /// Returns a list of all `Tool` instances contained within this ToolBox.
    fn tools(&self) -> Result<Vec<Arc<dyn Tool>>>;

    /// Retrieves a specific `Tool` from the ToolBox by its name.
    /// Returns `Some(tool)` if the tool is found, otherwise returns `None`.
    fn tool(&self, name: &str) -> Option<Arc<dyn Tool>>;
}
