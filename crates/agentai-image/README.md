# agentai-image

Image generation and editing tools for [AgentAI](https://crates.io/crates/agentai) agents.

`ImageToolBox` gives an agent the `generate_image` tool (and `edit_image`, when you tell it
where to load images from). The image is returned to the model as the tool result, and an
optional `ImageSink` saves every image, for example to a directory or your UI.

```rust,no_run
use agentai::Agent;
use agentai_image::ImageToolBox;

# async fn run() -> anyhow::Result<()> {
// The provider comes from the model name, the API key from `OPENAI_API_KEY`
let images = ImageToolBox::new("openai::gpt-image-1")?;
let agent = Agent::new("claude-haiku-5-5", "You are an illustrator.")?.with_toolbox(images);
let result = agent.run::<String>("Draw a lighthouse at dawn").await?;
# Ok(())
# }
```

Providers live in `agentai_image::provider`; `provider::openai` speaks the OpenAI image API
and any API compatible with it (`OPENAI_BASE_URL`, or a base URL of your own). See the
`image_generation` example for a sink and source that save images to a directory.
