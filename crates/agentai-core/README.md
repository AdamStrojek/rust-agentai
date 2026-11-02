# agentai-core

Shared building blocks of the [AgentAI](https://crates.io/crates/agentai) crates. You rarely
need it directly: it is a dependency of crates such as
[`agentai-image`](https://crates.io/crates/agentai-image).

It currently holds the HTTP client abstraction, `HttpClient`, so the crates can later run on
other HTTP stacks (for example in WebAssembly). `ReqwestClient` (feature `reqwest`, on by
default) implements it with `reqwest`.

TLS is chosen with features: `rustls-ring` (default), `rustls-aws-lc` or `native-tls` (with
`default-features = false`), plus `system-proxy` (default). With `rustls-ring`, call
`agentai_core::tls::install_crypto_provider()` before building your own `reqwest::Client`.
