use crate::tool::{Tool, ToolBox, ToolError, toolbox};
use anyhow::Context;
use reqwest::Client;
use serde_json::Value;

const BRAVE_API_URL: &str = "https://api.search.brave.com/res/v1/web/search";

/// # Brave Web Search Tool
///
/// This is a simple implementation of [crate::tool::ToolBox] for Web Search using Brave Search engine.
/// To use it you need to provide API Keys. This requires account creation, fortunately you can
/// choose free plan. Go to [<https://api.search.brave.com/app/keys>] to generate keys.
///
/// API Keys need to be provided when creating tool:
/// ```rust
///     let api_key = "<ENTER YOUR KEYS HERE>";
///     let tool = WebSearchToolBox::new(api_key);
/// ```
pub struct WebSearchToolBox {
    client: Client,
    api_key: String,
}

#[toolbox]
impl WebSearchToolBox {
    pub fn new(api_key: &str) -> Self {
        Self {
            client: Client::default(),
            api_key: api_key.to_string(),
        }
    }

    /// A tool that performs web searches using a specified query parameter to retrieve relevant
    /// results from a search engine. As the result you will receive list of websites with description
    #[tool]
    async fn web_search(
        &self,
        #[doc = "The search terms or keywords to be used by the search engine for retrieving relevant results"]
        query: String
    ) -> Result<String, ToolError> {
        let params = [("q", query.as_str()), ("count", "5"), ("result_filter", "web")];
        let response = self
            .client
            .get(BRAVE_API_URL)
            .query(&params)
            .header("X-Subscription-Token", self.api_key.clone())
            .send()
            .await.map_err(|e| anyhow::Error::new(e))?;

        let json: Value = response.json().await.map_err(|e| anyhow::Error::new(e))?;

        let mut results: Vec<String> = vec![];

        let response = json["web"]["results"].as_array().ok_or(ToolError::ExecutionError)?;
        for item in response
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
