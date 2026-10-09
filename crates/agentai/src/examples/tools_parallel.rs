//! # Search and fetch with Parallel Search MCP
//!
//! This example calls `web_search` and `web_fetch` through [`McpToolBox`](crate::tool::mcp::McpToolBox).
//! It uses the anonymous [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
//! endpoint: no Parallel API key or model credentials are required. Anonymous access
//! has lower rate limits and is intended for exploration and light use.
//!
//! ## Run
//!
//! Install [Node.js](https://nodejs.org/) 20 or later, including `npx`, and a stable
//! Rust toolchain. From the repository root, run:
//!
//! ```bash
//! cargo run --example tools_parallel
//! ```
//!
//! The first run downloads the pinned `mcp-remote@0.14.3` npm package. This bridge
//! connects AgentAI's stdio client to `https://search.parallel.ai/mcp` using
//! Streamable HTTP and honors `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY` when set. Keep the default `mcp-client` feature enabled.
//!
//! The example prints search results for Rust learning resources, then fetches
//! the Rust learning page. It calls tools directly, without running an LLM.
//!
//! ## Source code
//!
//! ```rust
#![doc = include_str!("../../examples/tools_parallel.rs")]
//! ```
