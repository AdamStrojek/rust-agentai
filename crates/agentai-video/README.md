# agentai-video

Video generation for [AgentAI](https://crates.io/crates/agentai).

Video APIs are asynchronous: `VideoProvider::submit` sends a request and returns a `VideoJob`, you follow it
with `VideoProvider::status` and, when it is `JobStatus::Completed`, fetch the video with
`VideoProvider::download`. Polling is up to you.

```rust,ignore
use std::time::Duration;
use agentai_video::{JobStatus, VideoRequest, from_model};

// The provider comes from the prefix of the model name, its settings from environment
// variables (`COMFY_SERVE_BASE_URL` here)
let provider = from_model("comfy-serve::wan-2.2")?;
let job = provider
    .submit(VideoRequest::new("A lighthouse at dawn").with_duration(5))
    .await?;
loop {
    match provider.status(&job).await? {
        JobStatus::Completed => break,
        JobStatus::Failed(message) => return Err(message.into()),
        JobStatus::Gone => return Err("the job is gone".into()),
        _ => tokio::time::sleep(Duration::from_secs(5)).await,
    }
}
let video = provider.download(&job).await?; // a `Media`, base64 or a URL
```

- A `VideoJob` can be stored (serde) and given back, possibly to a new provider, after a restart. A job does not
  record which provider made it, so store the provider and model name next to it.
- Three input modes in `VideoInputs`: none (text-to-video), `Frames { first, last }` (image-to-video) and
  `References(VideoReferences)` (reference images, videos and audio). A provider that cannot send an input
  returns `Unsupported`.
- Unset options (duration, resolution, aspect ratio, seed) are not sent and values are not checked here: the
  API reports the ones it does not accept.
- `list_models` returns `VideoModel`s with what a model accepts. `None` in a list means the provider does not
  say, not that the option is unsupported; `Some(vec![])` means there is nothing to choose.

| Prefix | Settings | Cancel | Inputs |
|--------|----------|--------|--------|
| `comfy-serve` | `COMFY_SERVE_BASE_URL` (required), `COMFY_SERVE_API_KEY` (optional) | yes | first and last frames; image, video and audio references; files are uploaded as multipart |
| `openrouter` | `OPENROUTER_API_KEY` (required), `OPENROUTER_BASE_URL` (optional) | no | first and last frames and image references, sent as URLs or data URLs |

The `openrouter` request format follows OpenRouter's public video generation guide and was not verified against
a live key.

To change timeouts or the proxy, build the provider yourself (`ComfyServeVideoProvider::new(model, api_key,
base_url)`) and give it your own client with `with_http_client(ReqwestClient::from_client(client))`.
`ReqwestClient`, `HttpClient` and the other HTTP types are re-exported from `agentai_video`, as are `Media` and
`mime`, so you need no direct `agentai-core` dependency. Implement `VideoProvider` for other APIs.

Files: `mime::from_path("first.jpg")` guesses the type of an input file for `Media::from_bytes`, and
`video.extension()` gives the file extension of a downloaded video's type (for example `mp4` or `webm`).

Limits: files travel in memory as `Media` (base64 or a URL); there is no waiting helper, no retries and no
timeouts of its own; there are no agent tools, the crate does not depend on `agentai`.

TLS features as in `agentai`: `rustls-ring` (default), `rustls-aws-lc`, `native-tls`, `system-proxy`.

See the `generate_video` example: it prints what the model accepts, takes an optional first frame and the
optional `AGENTAI_VIDEO_RESOLUTION` and `AGENTAI_VIDEO_DURATION` (seconds), and saves the video in `video/`.
