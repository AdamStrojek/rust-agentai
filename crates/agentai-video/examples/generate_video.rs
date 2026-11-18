//! Generates a video with `agentai-video`: submits a job, polls its status and saves the result
//! in `video/`.
//!
//! ```bash
//! # A comfy-serve server; the optional argument is an image used as the first frame
//! COMFY_SERVE_BASE_URL=http://localhost:8000 cargo run -p agentai-video --example generate_video
//! COMFY_SERVE_BASE_URL=http://localhost:8000 cargo run -p agentai-video --example generate_video -- first.png
//! # Optional: the resolution and the length in seconds, in the format of the provider
//! AGENTAI_VIDEO_MODEL=comfy-serve::wan-2.2 AGENTAI_VIDEO_RESOLUTION=720p AGENTAI_VIDEO_DURATION=5 \
//!     COMFY_SERVE_BASE_URL=http://localhost:8000 cargo run -p agentai-video --example generate_video
//! # OpenRouter, a model from `agentai_video::list_models("openrouter")`
//! OPENROUTER_API_KEY=<your key> AGENTAI_VIDEO_MODEL=openrouter::<model> \
//!     cargo run -p agentai-video --example generate_video
//! ```
//!
//! Before submitting, the example prints what the model accepts when the provider lists its
//! models. The crate has no waiting helper: the loop below is the polling you write yourself.

use std::time::Duration;

use agentai_video::{
    JobStatus, Media, ReqwestClient, VideoError, VideoInputs, VideoRequest, from_model, mime,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model =
        std::env::var("AGENTAI_VIDEO_MODEL").unwrap_or_else(|_| "comfy-serve::wan-2.2".to_string());
    let provider = from_model(&model)?;
    println!("Cancel supported: {}", provider.supports_cancel());

    // What the model accepts; `None` means the provider does not say
    match provider.list_models().await {
        Ok(models) => match models.iter().find(|m| m.model == model) {
            Some(m) => {
                println!("Durations: {:?}", m.durations);
                println!("Resolutions: {:?}", m.resolutions);
                println!("Aspect ratios: {:?}", m.aspect_ratios);
                println!("Inputs: {:?}", m.inputs);
            }
            None => println!("{model} is not in the provider's model list"),
        },
        Err(VideoError::Unsupported(_)) => {}
        Err(err) => println!("Could not list the models: {err}"),
    }

    let mut request = VideoRequest::new("A small lighthouse on a rock at dawn, waves rolling in");
    if let Ok(resolution) = std::env::var("AGENTAI_VIDEO_RESOLUTION") {
        request = request.with_resolution(resolution);
    }
    if let Ok(duration) = std::env::var("AGENTAI_VIDEO_DURATION") {
        request = request.with_duration(duration.parse()?);
    }
    if let Some(path) = std::env::args().nth(1) {
        let content_type = mime::from_path(&path)
            .ok_or_else(|| anyhow::anyhow!("unknown image type of {path}"))?;
        let first = Media::from_bytes(content_type, std::fs::read(&path)?);
        request = request.with_inputs(VideoInputs::Frames {
            first: Some(first),
            last: None,
        });
    }

    let job = provider.submit(request).await?;
    println!("Submitted job {}", job.id);

    let mut last = None;
    loop {
        let status = provider.status(&job).await?;
        if last.as_ref() != Some(&status) {
            println!("Status: {status:?}");
            last = Some(status.clone());
        }
        match status {
            JobStatus::Completed => break,
            JobStatus::Failed(message) => anyhow::bail!("The job failed: {message}"),
            JobStatus::Gone => anyhow::bail!("The provider no longer knows the job"),
            _ => tokio::time::sleep(Duration::from_secs(5)).await,
        }
    }

    let video = provider.download(&job).await?;
    // A base64 video is decoded, a URL is downloaded
    let bytes = video.bytes(&ReqwestClient::new()).await?;
    std::fs::create_dir_all("video")?;
    let path = format!("video/{}.{}", job.id, video.extension());
    std::fs::write(&path, bytes)?;
    println!("Saved {path}");
    Ok(())
}
