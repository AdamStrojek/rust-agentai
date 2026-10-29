# agentai-core

Shared building blocks of the [AgentAI](https://crates.io/crates/agentai) crates. You rarely
need it directly: it is a dependency of crates such as
[`agentai-image`](https://crates.io/crates/agentai-image).

It currently holds the HTTP client abstraction, `HttpClient`, so the crates can later run on
other HTTP stacks (for example in WebAssembly). `ReqwestClient` (feature `reqwest`, on by
default) implements it with `reqwest`.
