# agentai-core

Shared building blocks of the [AgentAI](https://crates.io/crates/agentai) crates. You rarely
need it directly: it is a dependency of crates such as
[`agentai-image`](https://crates.io/crates/agentai-image).

It holds the conversation types (`Message`, `Media`, ...), media storage, and the HTTP client
abstraction, `HttpClient`, so the crates can later run on other HTTP stacks (for example in
WebAssembly). `ReqwestClient` (feature `reqwest`, on by default) implements it with `reqwest`.

What the media crates share lives here too: `mime` maps MIME types to file extensions and back
(`Media::extension`, `Media::file_name`), `base64` encodes and decodes media bytes
(`Media::to_url` gives a data URL), and the `test-support` feature adds `mock::MockHttp`, a
recording `HttpClient` for tests of providers.

TLS is chosen with features: `rustls-ring` (default), `rustls-aws-lc` or `native-tls` (with
`default-features = false`), plus `system-proxy` (default). With `rustls-ring`, call
`agentai_core::tls::install_crypto_provider()` before building your own `reqwest::Client`.
