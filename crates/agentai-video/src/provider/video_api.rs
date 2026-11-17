//! Reading the replies of the "Video API v1" protocol family, shared by the providers that
//! speak it.

use agentai_core::HttpResponse;
use agentai_core::message::Media;
use serde_json::Value;

use super::{JobStatus, VideoError, VideoInputKind, VideoJob, VideoModel};

fn json_body(response: &HttpResponse) -> Result<Value, VideoError> {
    response
        .json()
        .map_err(|err| VideoError::InvalidResponse(err.to_string()))
}

/// `{ "id", "polling_url" }` becomes `VideoJob { id, data: {"polling_url": ..} }`.
pub(crate) fn parse_job(response: &HttpResponse) -> Result<VideoJob, VideoError> {
    let json = json_body(response)?;
    let id = json["id"]
        .as_str()
        .ok_or_else(|| VideoError::InvalidResponse(format!("no job id in {json}")))?;
    Ok(VideoJob::new(id).with_data(serde_json::json!({ "polling_url": json["polling_url"] })))
}

/// 404 is `Gone`; otherwise the status field is mapped.
pub(crate) fn parse_status(response: HttpResponse) -> Result<JobStatus, VideoError> {
    if response.status == 404 {
        return Ok(JobStatus::Gone);
    }
    let json = json_body(&response.error_for_status()?)?;
    match json["status"].as_str() {
        Some("pending") => Ok(JobStatus::Queued),
        Some("in_progress") => Ok(JobStatus::Running),
        Some("completed") => Ok(JobStatus::Completed),
        Some("cancelled" | "expired") => Ok(JobStatus::Gone),
        Some("failed") => {
            let error = &json["error"];
            let message = error["message"]
                .as_str()
                .or_else(|| error.as_str())
                .unwrap_or("the job failed");
            Ok(JobStatus::Failed(message.to_string()))
        }
        other => Err(VideoError::InvalidResponse(format!(
            "unknown job status {other:?}"
        ))),
    }
}

/// `data[]` becomes the models; their names are `{prefix}::{id}`.
pub(crate) fn parse_models(
    prefix: &str,
    response: &HttpResponse,
) -> Result<Vec<VideoModel>, VideoError> {
    let json = json_body(response)?;
    let data = json["data"]
        .as_array()
        .ok_or_else(|| VideoError::InvalidResponse("no `data` list of models".to_string()))?;
    Ok(data
        .iter()
        .filter_map(|entry| {
            let id = entry["id"].as_str()?;
            let mut model = VideoModel::new(format!("{prefix}::{id}"), id);
            model.name = entry["name"].as_str().map(str::to_string);
            model.durations = list(&entry["supported_durations"], |v| {
                v.as_u64().and_then(|n| u32::try_from(n).ok())
            })
            .map(|mut durations| {
                durations.sort_unstable();
                durations
            });
            model.resolutions = list(&entry["supported_resolutions"], |v| {
                v.as_str().map(str::to_string)
            });
            model.aspect_ratios = list(&entry["supported_aspect_ratios"], |v| {
                v.as_str().map(str::to_string)
            });
            model.inputs = list(&entry["supported_inputs"], |v| match v.as_str()? {
                "first_frame" => Some(VideoInputKind::FirstFrame),
                "last_frame" => Some(VideoInputKind::LastFrame),
                "image" => Some(VideoInputKind::Image),
                "video" => Some(VideoInputKind::Video),
                "audio" => Some(VideoInputKind::Audio),
                _ => None,
            });
            Some(model)
        })
        .collect())
}

/// The items of a JSON array that `item` accepts; `None` when the key is not an array.
fn list<T>(value: &Value, item: impl Fn(&Value) -> Option<T>) -> Option<Vec<T>> {
    Some(value.as_array()?.iter().filter_map(item).collect())
}

/// The video bytes, typed by the `Content-Type` header (`video/mp4` without one).
pub(crate) fn parse_content(response: HttpResponse) -> Result<Media, VideoError> {
    let response = response.error_for_status()?;
    let content_type = response
        .header("content-type")
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("video/mp4")
        .to_string();
    Ok(Media::from_bytes(content_type, response.body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::tests::json_response;
    use serde_json::json;

    #[test]
    fn statuses_map_to_job_status() {
        let status = |code, body: Value| parse_status(json_response(code, body));
        assert_eq!(
            status(200, json!({"status": "pending"})).unwrap(),
            JobStatus::Queued
        );
        assert_eq!(
            status(200, json!({"status": "in_progress"})).unwrap(),
            JobStatus::Running
        );
        assert_eq!(
            status(200, json!({"status": "completed"})).unwrap(),
            JobStatus::Completed
        );
        for gone in ["cancelled", "expired"] {
            assert_eq!(
                status(200, json!({"status": gone})).unwrap(),
                JobStatus::Gone
            );
        }
        assert_eq!(
            status(404, json!({"error": {"message": "no job"}})).unwrap(),
            JobStatus::Gone
        );
        assert_eq!(
            status(
                200,
                json!({"status": "failed", "error": {"message": "oom"}})
            )
            .unwrap(),
            JobStatus::Failed("oom".into())
        );
        assert_eq!(
            status(200, json!({"status": "failed", "error": "bad prompt"})).unwrap(),
            JobStatus::Failed("bad prompt".into())
        );
        assert!(
            matches!(status(200, json!({"status": "failed"})).unwrap(), JobStatus::Failed(m) if !m.is_empty())
        );
        assert!(matches!(
            status(200, json!({"status": "dreaming"})),
            Err(VideoError::InvalidResponse(_))
        ));
        assert!(matches!(
            status(500, json!({"error": {"message": "boom"}})),
            Err(VideoError::Api { status: 500, message }) if message == "boom"
        ));
    }

    #[test]
    fn a_job_without_an_id_is_an_invalid_response() {
        assert!(matches!(
            parse_job(&json_response(202, json!({"polling_url": "/x"}))),
            Err(VideoError::InvalidResponse(_))
        ));
        let job = parse_job(&json_response(
            202,
            json!({"id": "j1", "polling_url": "/api/v1/videos/j1"}),
        ))
        .unwrap();
        assert_eq!(
            job,
            VideoJob::new("j1").with_data(json!({"polling_url": "/api/v1/videos/j1"}))
        );
    }

    #[test]
    fn models_keep_absent_capabilities_unknown() {
        let models = parse_models(
            "comfy-serve",
            &json_response(
                200,
                json!({"data": [
                    {"id": "wan", "name": "Wan", "supported_durations": [10, 5],
                     "supported_resolutions": ["720p"], "supported_aspect_ratios": ["16:9"],
                     "supported_inputs": ["first_frame", "audio", "hologram"]},
                    {"id": "bare", "supported_durations": []},
                ]}),
            ),
        )
        .unwrap();
        assert_eq!(models[0].model, "comfy-serve::wan");
        assert_eq!(models[0].id, "wan");
        assert_eq!(models[0].durations, Some(vec![5, 10]));
        assert_eq!(
            models[0].inputs,
            Some(vec![VideoInputKind::FirstFrame, VideoInputKind::Audio])
        );
        assert_eq!(models[1].durations, Some(vec![]));
        assert_eq!(
            (
                models[1].resolutions.clone(),
                models[1].inputs.clone(),
                models[1].name.clone()
            ),
            (None, None, None)
        );
        assert!(matches!(
            parse_models("x", &json_response(200, json!({}))),
            Err(VideoError::InvalidResponse(_))
        ));
    }

    #[test]
    fn content_type_comes_from_the_header_or_defaults_to_mp4() {
        let media = parse_content(
            HttpResponse::new(200, vec![1u8, 2])
                .with_header("Content-Type", "video/webm; codecs=vp9"),
        )
        .unwrap();
        assert_eq!(media.content_type, "video/webm");
        assert_eq!(
            parse_content(HttpResponse::new(200, vec![1u8]))
                .unwrap()
                .content_type,
            "video/mp4"
        );
    }
}
