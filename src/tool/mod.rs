pub mod websearch;

use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait AgentTool<CTX> {
    fn name(&self) -> String;

    fn description(&self) -> String;

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

    async fn call(&self, context: &CTX, params: Value) -> anyhow::Result<String>;
}
