use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FileRecord {
    pub path: String,
    pub name: String,
    pub extension: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: u64,
    pub content_embedding: Vec<f32>,
    pub content_preview: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResult {
    pub file: FileRecord,
    pub score: f32,
}