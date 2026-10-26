//! # Vision Example
//!
//! This example sends the project logo to a model that understands images. A
//! [`Prompt`](crate::Prompt) combines text with media: first the logo is read from the
//! repository with `include_bytes!` and sent as base64 with
//! [`Prompt::with_image_bytes`](crate::Prompt::with_image_bytes), then the same file is passed
//! by its URL on GitHub. Not every provider accepts image URLs (local servers such as oMLX
//! take only base64 data); the example shows how to recognise that error.
//!
//! The model has to support images. A model without vision support usually answers with an
//! error.
//!
//! To run this example from the terminal, enter:
//! ```bash
//! cargo run --example vision
//! ```
//!
//! ## Source Code
//!
//! ```rust
#![doc = include_str!("../../examples/vision.rs")]
//! ```
