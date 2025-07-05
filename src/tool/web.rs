//! # Web Tools
//!
//! This module provides a collection of tools designed for interacting with the web.
//! It includes functionalities such as performing web searches and fetching content from URLs.
//! These tools empower an AI agent to access and process information from the internet.
//!
//! For a practical demonstration of these tools, please refer to the example located at
//! [examples/tools_web.rs](crate::examples::tools_web).

use crate::tool::{Tool, ToolBox, ToolError, ToolResult, toolbox};
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
    /// Creates a new instance of `WebSearchToolBox`.
    ///
    /// # Arguments
    ///
    /// * `api_key` - A string slice that holds the API key for the Brave Search API.
    pub fn new(api_key: &str) -> Self {
        Self {
            client: Client::default(),
            api_key: api_key.to_string(),
        }
    }

    /// A tool that performs web searches using a specified query parameter to retrieve relevant
    /// results from a search engine. As the result you will receive list of websites with description.
    ///
    /// ## Example
    ///
    /// **User:** "What is the latest news about AI?"
    /// ```
    #[tool]
    pub async fn web_search(
        &self,
        #[doc = "The search terms or keywords to be used by the search engine for retrieving relevant results."]
        query: String,
    ) -> ToolResult {
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

/// Provides a tool that enables an LLM to fetch the content of a web page.
/// This is useful for accessing the raw text from a website to be used as context.
pub struct WebFetchToolBox {
    client: Client,
}

impl Default for WebFetchToolBox {
    fn default() -> Self {
        Self::new()
    }
}

#[toolbox]
impl WebFetchToolBox {
    /// Creates a new instance of `WebFetchToolBox`.
    pub fn new() -> Self {
        Self {
            client: Client::default(),
        }
    }

    /// Fetches the content of a web page given its URL. This tool is useful for accessing the
    /// raw text content of a webpage. The content is returned as a single string.
    ///
    /// ## Example
    ///
    /// **User:** "Fetch me page at: https://github.com/AdamStrojek/rust-agentai/"
    #[tool]
    pub async fn web_fetch(
        &self,
        #[doc = "The full URL of the web page to fetch, including the protocol (e.g., https://)."]
        url: String
    ) -> ToolResult {
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ToolError::LLMError(format!("Request to {} failed: {}", url, e)))?;

        if !response.status().is_success() {
            return Err(ToolError::LLMError(format!(
                "Request to {} failed with status: {}",
                url,
                response.status()
            )));
        }

        let body = response.text().await.map_err(|e| anyhow::Error::new(e))?;

        Ok(body)
    }
}
