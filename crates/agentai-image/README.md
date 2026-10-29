# agentai-image

Image generation and editing for [AgentAI](https://crates.io/crates/agentai).

- `ImageGenerator` generates and edits images from your code.
- With the `toolbox` feature (on by default), `GenerateImageToolBox` gives an agent the
  `generate_image` and `edit_image` tools. The model gets the image, or only its metadata as
  JSON. `ReadMediaToolBox` lets the model read a stored file by its ID.
- Images are saved in any `MediaStorage` from `agentai-core` (`MediaRead` + `MediaSave`).

```rust,ignore
use agentai_image::ImageGenerator;

// The provider comes from the model name, the API key from `OPENAI_API_KEY`
let generator = ImageGenerator::new("openai::gpt-image-1")?.with_size("1024x1024");
let response = generator.generate("A lighthouse at dawn").await?;
```

Providers live in `agentai_image::provider`. `provider::openai` speaks the OpenAI image API and
any API compatible with it (`OPENAI_BASE_URL`, or a base URL set in code). See the
`image_generator` and `image_tools` examples.
