//! Storage for media: images, audio and other files, kept under an ID.
//!
//! [`MediaSave`] stores a [`Media`] and returns its ID, [`MediaRead`] returns it by that ID.
//! A type with both is a [`MediaStorage`]. Implement them for your blob store, database or
//! directory; [`MemoryStorage`] keeps the media in memory.

use crate::message::Media;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Errors of a media storage.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StorageError {
    /// Nothing is stored under this ID.
    #[error("no media with id '{0}'")]
    NotFound(String),
    /// Any other failure, for example of the disk or the network.
    #[error(transparent)]
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl StorageError {
    /// Creates a [`StorageError::Other`] from any error.
    pub fn other(err: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Other(err.into())
    }
}

/// Reads media by ID.
#[async_trait]
pub trait MediaRead: Send + Sync + 'static {
    /// Returns the media stored under `id`, or [`StorageError::NotFound`].
    async fn read(&self, id: &str) -> Result<Media, StorageError>;
}

/// Saves media and chooses its ID.
#[async_trait]
pub trait MediaSave: Send + Sync + 'static {
    /// Stores `media` and returns the ID to read it with. The storage chooses the ID; it may
    /// use [`Media::name`].
    async fn save(&self, media: &Media) -> Result<String, StorageError>;
}

/// A storage that can save and read media. Implemented for every type with both traits.
pub trait MediaStorage: MediaRead + MediaSave {}

impl<T: MediaRead + MediaSave> MediaStorage for T {}

#[async_trait]
impl<T: MediaRead + ?Sized> MediaRead for Arc<T> {
    async fn read(&self, id: &str) -> Result<Media, StorageError> {
        (**self).read(id).await
    }
}

#[async_trait]
impl<T: MediaSave + ?Sized> MediaSave for Arc<T> {
    async fn save(&self, media: &Media) -> Result<String, StorageError> {
        (**self).save(media).await
    }
}

/// [`MediaStorage`] in memory. The ID is the media's name with a number in front
/// (`1-cat.png`), or `media-1` for media without a name.
///
/// Clones share the same storage.
#[derive(Debug, Clone, Default)]
pub struct MemoryStorage {
    inner: Arc<Mutex<MemoryInner>>,
}

#[derive(Debug, Default)]
struct MemoryInner {
    count: usize,
    media: HashMap<String, Media>,
}

impl MemoryStorage {
    /// Creates an empty storage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the IDs of the stored media.
    pub fn ids(&self) -> Vec<String> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.media.keys().cloned().collect()
    }
}

#[async_trait]
impl MediaRead for MemoryStorage {
    async fn read(&self, id: &str) -> Result<Media, StorageError> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner
            .media
            .get(id)
            .cloned()
            .ok_or_else(|| StorageError::NotFound(id.to_string()))
    }
}

#[async_trait]
impl MediaSave for MemoryStorage {
    async fn save(&self, media: &Media) -> Result<String, StorageError> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.count += 1;
        let id = match &media.name {
            Some(name) => format!("{}-{name}", inner.count),
            None => format!("media-{}", inner.count),
        };
        inner.media.insert(id.clone(), media.clone());
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_storage_reads_back_what_was_saved_under_distinct_ids() {
        let storage = MemoryStorage::new();
        let cat = Media::from_bytes("image/png", b"cat").with_name("a.png");
        let dog = Media::from_bytes("image/png", b"dog").with_name("a.png");

        let cat_id = storage.save(&cat).await.unwrap();
        let dog_id = storage.save(&dog).await.unwrap();
        let unnamed = storage
            .save(&Media::from_bytes("image/png", b"x"))
            .await
            .unwrap();

        assert_ne!(cat_id, dog_id);
        assert_eq!(unnamed, "media-3");
        assert_eq!(storage.read(&cat_id).await.unwrap(), cat);
        assert_eq!(storage.read(&dog_id).await.unwrap(), dog);
        assert!(matches!(
            storage.read("missing").await,
            Err(StorageError::NotFound(id)) if id == "missing"
        ));
    }
}
