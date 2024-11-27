use thiserror::Error;

#[derive(Error, Debug)]
pub enum FileEmbeddingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Database error: {0}")]
    Database(#[from] surrealdb::Error),
    #[error("Embedding error: {0}")]
    Embedding(String),
    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),
    #[error("PDF extraction error: {0}")]
    PdfExtraction(String),
    #[error("WalkDir error: {0}")]
    WalkDir(#[from] walkdir::Error),
}
