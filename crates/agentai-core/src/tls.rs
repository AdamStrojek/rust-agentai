//! TLS backend.
//!
//! The backend is chosen with features of the AgentAI crates:
//!
//! | Feature | Backend |
//! | --- | --- |
//! | `rustls-ring` (default) | rustls with the ring crypto provider, OS trust store |
//! | `rustls-aws-lc` | rustls with the aws-lc-rs crypto provider, OS trust store |
//! | `native-tls` | TLS of the operating system; needs `default-features = false` |
//!
//! Without any of them HTTPS requests fail, unless you bring a `reqwest` client with TLS
//! yourself.
//!
//! With `rustls-ring`, rustls needs ring installed as the process-wide crypto provider before
//! a `reqwest` client is built, or building it panics. The AgentAI crates do this whenever they
//! build a client themselves. Call [`install_crypto_provider`] first when you build your own
//! `reqwest::Client` or `genai::Client`.

/// Installs the crypto provider chosen by the `rustls-ring` feature as the process default.
///
/// It does nothing when a provider is already installed (yours wins) or when the feature is
/// off, so it is safe to call any number of times.
pub fn install_crypto_provider() {
    #[cfg(feature = "rustls-ring")]
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        // Fails only when another thread installed a provider in the meantime.
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
}
