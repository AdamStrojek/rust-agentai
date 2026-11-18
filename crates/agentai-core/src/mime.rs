//! MIME types and file extensions of images, audio and video.
//!
//! One table for every crate: [`extension`] names a file for a type, [`from_path`] guesses the
//! type of a file name or URL, and [`with_extension`] gives a name the extension of its type.

/// Extensions [`from_path`] knows and the MIME type each one stands for.
const TYPES: &[(&str, &str)] = &[
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
    ("avif", "image/avif"),
    ("bmp", "image/bmp"),
    ("svg", "image/svg+xml"),
    ("mp3", "audio/mpeg"),
    ("wav", "audio/wav"),
    ("opus", "audio/ogg"),
    ("ogg", "audio/ogg"),
    ("flac", "audio/flac"),
    ("aac", "audio/aac"),
    ("pcm", "audio/pcm"),
    ("m4a", "audio/mp4"),
    ("mp4", "video/mp4"),
    ("webm", "video/webm"),
    ("mov", "video/quicktime"),
    ("mkv", "video/x-matroska"),
    ("avi", "video/x-msvideo"),
];

/// File extension for a MIME type, without the dot. Parameters (`; codecs=...`) are ignored.
///
/// `audio/mpeg` is `mp3`, `audio/ogg` is `opus`, `video/quicktime` is `mov`,
/// `image/svg+xml` is `svg`, and the common aliases have their usual names (`audio/x-wav` is
/// `wav`, `audio/mp4` is `m4a`, `video/x-matroska` is `mkv`, `video/x-msvideo` is `avi`); any
/// other type gives its subtype when that is only ASCII letters
/// and digits (`image/png` is `png`), and `bin` otherwise.
///
/// ```rust
/// use agentai_core::mime;
/// assert_eq!(mime::extension("audio/mpeg"), "mp3");
/// assert_eq!(mime::extension("video/webm; codecs=vp9"), "webm");
/// assert_eq!(mime::extension("application/octet-stream"), "bin");
/// ```
pub fn extension(content_type: &str) -> &str {
    let essence = content_type.split(';').next().unwrap_or_default().trim();
    match essence.to_ascii_lowercase().as_str() {
        "audio/mpeg" => return "mp3",
        "audio/ogg" => return "opus",
        "audio/x-wav" | "audio/wave" | "audio/vnd.wave" => return "wav",
        "audio/mp4" | "audio/x-m4a" => return "m4a",
        "video/x-matroska" => return "mkv",
        "video/x-msvideo" => return "avi",
        "video/quicktime" => return "mov",
        "image/svg+xml" => return "svg",
        _ => {}
    }
    essence
        .split_once('/')
        .map(|(_, subtype)| subtype)
        .filter(|subtype| !subtype.is_empty() && subtype.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or("bin")
}

/// MIME type of a file name, path or URL from its extension, ignoring case, a `?query` and a
/// `#fragment`. `None` when the extension is not an image, audio or video type this module
/// knows.
///
/// ```rust
/// use agentai_core::mime;
/// assert_eq!(mime::from_path("clips/intro.MP4"), Some("video/mp4"));
/// assert_eq!(mime::from_path("https://example.com/a.jpg?size=2"), Some("image/jpeg"));
/// assert_eq!(mime::from_path("notes.txt"), None);
/// ```
pub fn from_path(path_or_url: &str) -> Option<&'static str> {
    let path = path_or_url.split(['?', '#']).next().unwrap_or_default();
    let file = path.rsplit(['/', '\\']).next().unwrap_or_default();
    let (_, extension) = file.rsplit_once('.')?;
    TYPES
        .iter()
        .find(|(known, _)| known.eq_ignore_ascii_case(extension))
        .map(|(_, content_type)| *content_type)
}

/// `name` as it is when it ends in an extension [`from_path`] knows, otherwise `name` with the
/// [`extension`] of `content_type` added.
///
/// ```rust
/// use agentai_core::mime;
/// assert_eq!(mime::with_extension("intro.wav", "audio/mpeg"), "intro.wav");
/// assert_eq!(mime::with_extension("intro.v2", "audio/mpeg"), "intro.v2.mp3");
/// ```
pub fn with_extension(name: &str, content_type: &str) -> String {
    if from_path(name).is_some() {
        name.to_string()
    } else {
        format!("{name}.{}", extension(content_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_follows_the_subtype_with_a_few_exceptions() {
        for (content_type, expected) in [
            ("image/png", "png"),
            ("image/jpeg; q=1", "jpeg"),
            ("audio/webm;codecs=opus", "webm"),
            ("audio/mpeg; charset=x", "mp3"),
            ("audio/ogg", "opus"),
            ("video/quicktime", "mov"),
            ("image/svg+xml", "svg"),
            ("application/octet-stream", "bin"),
            ("audio/x-wav", "wav"),
            ("video/x-matroska", "mkv"),
            ("application/x-unknown", "bin"),
            ("video", "bin"),
            ("", "bin"),
        ] {
            assert_eq!(extension(content_type), expected, "{content_type}");
        }
    }

    #[test]
    fn from_path_reads_the_last_extension_of_the_file() {
        for (path, expected) in [
            ("a.PNG", Some("image/png")),
            (
                "https://example.com/v.webm?x=1.png#t.gif",
                Some("video/webm"),
            ),
            ("dir.mp4/file", None),
            ("voice.opus", Some("audio/ogg")),
            ("notes.txt", None),
            ("noextension", None),
        ] {
            assert_eq!(from_path(path), expected, "{path}");
        }
    }

    #[test]
    fn with_extension_keeps_only_known_extensions() {
        for (name, content_type, expected) in [
            ("intro", "audio/mpeg", "intro.mp3"),
            ("intro.v2", "audio/mpeg", "intro.v2.mp3"),
            ("intro.wav", "audio/mpeg", "intro.wav"),
            ("my.picture", "image/png", "my.picture.png"),
        ] {
            assert_eq!(with_extension(name, content_type), expected);
        }
    }
}
