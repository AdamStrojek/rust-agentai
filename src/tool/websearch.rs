use crate::AgentTool;
use anyhow::Context;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

const BRAVE_API_URL: &str = "https://api.search.brave.com/res/v1/web/search";

/// # Brave Web Search Tool
///
/// This is simple implementation of [crate::AgentTool] for Web Search using Brave Search engine.
/// To use it you need to provide API Keys. This requires account creation, fortunetly you can
/// choose free plan. Go to [https://api.search.brave.com/app/keys] to generate keys.
///
/// API Keys need to be provided when creating tool:
/// ```rust
///     let api_key = "<ENTER YOUR KEYS HERE>";
///     let tool = WebSearchTool::new(api_key);
/// ```
pub struct WebSearchTool {
    client: Client,
    api_key: String,
}

impl WebSearchTool {
    pub fn new(api_key: &str) -> Self {
        Self {
            client: Client::default(),
            api_key: api_key.to_string(),
        }
    }
}

#[async_trait]
impl<CTX> AgentTool<CTX> for WebSearchTool {
    fn name(&self) -> String {
        "Web_Search".to_string()
    }

    fn description(&self) -> String {
        "Use this function to perform websearch. As the result you will receive list of websites with description".to_string()
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                  "description": "The unique identifier for a product",
                  "type": "string"
                },
            },
            "required": [ "query", "productName", "price" ]
        })
    }

    async fn call(&self, _: &CTX, params: Value) -> anyhow::Result<String> {
        let query = params["query"]
            .as_str()
            .expect("Query parameter is not a string");

        let params = [("q", query), ("count", "5"), ("result_filter", "web")];
        let response = self
            .client
            .get(BRAVE_API_URL)
            .query(&params)
            .header("X-Subscription-Token", self.api_key.clone())
            .send()
            .await?;

        let json: Value = response.json().await?;

        let mut results: Vec<String> = vec![];

        for item in json["web"]["results"]
            .as_array()
            .context("web results is not an array")?
        {
            let title = item["title"]
                .as_str()
                .context("web title is not a string")?;
            let description = item["description"]
                .as_str()
                .context("web description is not a string")?;
            let url = item["url"].as_str().context("web url is not a string")?;
            results.push(format!(
                "Title: {title}\nDescription: {description}\nURL: {url}"
            ));
        }

        Ok(results.join("\n\n"))
    }
}
