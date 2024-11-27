use anyhow::Result;
use fastembed::{ TextEmbedding, InitOptions, EmbeddingModel };
use std::path::PathBuf;
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use walkdir::WalkDir;
use std::fs;
mod error;
mod models;
use models::{ FileRecord, SearchResult };
use error::FileEmbeddingError;
use serde::{ Deserialize, Serialize };
use surrealdb::opt::RecordId;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum();
    let magnitude_a: f32 = a
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    let magnitude_b: f32 = b
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

const SUPPORTED_TEXT_EXTENSIONS: &[&str] = &[
    "txt",
    "md",
    "rs",
    "py",
    "js",
    "json",
    "yaml",
    "yml",
    "toml",
    "css",
    "html",
    "htm",
    "xml",
    "csv",
    "log",
    "pdf",
    "doc",
    "docx",
];

pub struct FileEmbeddingSystem {
    db: Surreal<surrealdb::engine::local::Db>,
    embedding_model: TextEmbedding,
}

#[derive(Debug, Deserialize, Clone)]
struct Record {
    #[allow(dead_code)]
    id: RecordId,
}

impl FileEmbeddingSystem {
    pub async fn new(db_path: &str) -> Result<Self> {
        // Initialize SurrealDB with RocksDB
        let db = Surreal::new::<RocksDb>(db_path).await?;
        db.query("REMOVE TABLE files").await?;
        db.use_ns("files").use_db("embeddings").await?;
        // Update schema definition
        db.query(
            "
            DEFINE TABLE files SCHEMAFUL;
            DEFINE FIELD path ON files TYPE string;
            DEFINE FIELD name ON files TYPE string;
            DEFINE FIELD extension ON files TYPE option<string>;
            DEFINE FIELD mime_type ON files TYPE option<string>;
            DEFINE FIELD size_bytes ON files TYPE number;
            DEFINE FIELD content_embedding ON files TYPE array<float>;
            DEFINE FIELD content_preview ON files TYPE string;
            
            DEFINE INDEX idx_path ON files FIELDS path UNIQUE;
            DEFINE INDEX idx_name ON files FIELDS name;
            DEFINE INDEX idx_extension ON files FIELDS extension;
        "
        ).await?;

        // Initialize FastEmbed model
        let embedding_model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true)
        )?;

        Ok(Self {
            db,
            embedding_model,
        })
    }

    async fn extract_text_content(&self, path: &PathBuf) -> Result<String, FileEmbeddingError> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "pdf" => {
                let bytes = fs::read(path)?;
                pdf_extract
                    ::extract_text_from_mem(&bytes)
                    .map_err(|e| FileEmbeddingError::PdfExtraction(e.to_string()))
            }
            _ if SUPPORTED_TEXT_EXTENSIONS.contains(&extension.as_str()) => {
                fs::read_to_string(path).map_err(FileEmbeddingError::Io)
            }
            _ => Err(FileEmbeddingError::UnsupportedFileType(extension)),
        }
    }

    pub async fn index_file(&self, path: PathBuf) -> Result<(), FileEmbeddingError> {
        let metadata = fs::metadata(&path)?;
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        println!("Attempting to index: {}", path.display());

        if
            !metadata.is_file() ||
            !extension
                .as_ref()
                .map_or(false, |ext| SUPPORTED_TEXT_EXTENSIONS.contains(&ext.as_str()))
        {
            println!("Skipping unsupported file: {}", path.display());
            return Err(
                FileEmbeddingError::UnsupportedFileType(
                    extension.unwrap_or_else(|| String::from("unknown")).clone()
                )
            );
        }

        match self.extract_text_content(&path).await {
            Ok(content) => {
                println!("Successfully extracted content from: {}", path.display());
                let embeddings = self.embedding_model
                    .embed(vec![content.clone()], None)
                    .map_err(|e| FileEmbeddingError::Embedding(e.to_string()))?;

                println!("Generated embedding with size: {}", embeddings[0].len());

                let content_preview = content.chars().take(1000).collect::<String>();

                let file_record = FileRecord {
                    path: path.to_string_lossy().to_string(),
                    name: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                    extension,
                    mime_type: mime_guess
                        ::from_path(&path)
                        .first()
                        .map(|m| m.to_string()),
                    size_bytes: metadata.len(),
                    content_embedding: embeddings[0].clone(),
                    content_preview,
                };

                // Debug: Print sample of embedding before storage
                println!(
                    "First few values of embedding for {}: {:?}",
                    file_record.name,
                    file_record.content_embedding.iter().take(5).collect::<Vec<_>>()
                );

                // Store in database
                let created: Option<FileRecord> = self.db
                    .create("files")
                    .content(file_record).await?
                    .into_iter()
                    .next();

                if let Some(record) = created {
                    println!(
                        "Successfully indexed: {} (embedding size: {})",
                        path.display(),
                        record.content_embedding.len()
                    );
                } else {
                    println!("Warning: File created but no record returned: {}", path.display());
                }

                Ok(())
            }
            Err(e) => {
                println!("Error extracting content from {}: {:?}", path.display(), e);
                Err(e)
            }
        }
    }

    pub async fn index_directory(&self, dir_path: PathBuf) -> Result<(), FileEmbeddingError> {
        for entry in WalkDir::new(dir_path) {
            let entry = entry?;
            if entry.file_type().is_file() {
                if let Err(e) = self.index_file(entry.path().to_path_buf()).await {
                    eprintln!("Error indexing {}: {:?}", entry.path().display(), e);
                }
            }
        }
        Ok(())
    }
    pub async fn hybrid_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embedding_model.embed(vec![query.to_string()], None)?[0].clone();
        println!("Query embedding size: {}", query_embedding.len());

        // Get all records first to help debug
        let all_records: Vec<FileRecord> = self.db.query("SELECT * FROM files").await?.take(0)?;
        println!("Total records in DB: {}", all_records.len());

        // Compute similarities in Rust instead of relying on SurrealDB's vector operations
        let mut results: Vec<SearchResult> = all_records
            .into_iter()
            .map(|record| {
                let similarity = cosine_similarity(&record.content_embedding, &query_embedding);
                SearchResult {
                    file: record,
                    score: similarity,
                }
            })
            .collect();

        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take only the requested number of results
        results.truncate(limit);

        println!("Found {} results", results.len());
        for result in results.iter() {
            println!("Path: {}, Score: {}", result.file.path, result.score);
        }

        Ok(results)
    }
}
#[tokio::main]
async fn main() -> Result<()> {
    let system = FileEmbeddingSystem::new("./db").await?;

    // Example: Index files from Desktop
    let desktop = dirs::desktop_dir().expect("Failed to get desktop directory");

    println!("Indexing files from Desktop...");
    system.index_directory(desktop).await?;
    println!("Indexing complete!");

    // Example: Perform a search
    println!("\nSearching for 'rust programming'...");
    let results = system.hybrid_search("rust programming", 5).await?;

    if results.is_empty() {
        println!("No results found!");
    } else {
        println!("\nSearch Results:");
        println!("---------------");
        for (i, result) in results.iter().enumerate() {
            println!("{}. File: {}", i + 1, result.file.name);
            println!("   Path: {}", result.file.path);
            println!("   Score: {:.4}", result.score);
            println!("   Preview: {:.100}...", result.file.content_preview);
            println!();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir; // Add tempfile to your dependencies

    async fn setup_test_system() -> (FileEmbeddingSystem, TempDir) {
        // Create a temporary directory for the database
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("test_db");

        let system = FileEmbeddingSystem::new(db_path.to_str().unwrap()).await.expect(
            "Failed to create FileEmbeddingSystem"
        );

        (system, temp_dir)
    }

    #[tokio::test]
    async fn test_file_indexing_and_search() -> Result<()> {
        let (system, _temp_dir) = setup_test_system().await;

        // Create a test file
        let test_dir = TempDir::new()?;
        let test_file_path = test_dir.path().join("test.txt");

        let test_content =
            r#"Rust is a systems programming language that runs blazingly fast, prevents segfaults, 
        and guarantees thread safety. Features include zero-cost abstractions, move semantics,
        guaranteed memory safety, threads without data races, trait-based generics,
        pattern matching, type inference, minimal runtime, and efficient C bindings."#;

        fs::write(&test_file_path, test_content)?;

        // Test file indexing
        system.index_file(test_file_path).await?;

        // Test search functionality
        let search_terms = [
            ("rust programming", true),
            ("memory safety", true),
            ("python scripting", false), // Should not match strongly
        ];

        for (term, should_match) in search_terms {
            let results = system.hybrid_search(term, 5).await?;

            if should_match {
                assert!(!results.is_empty(), "Expected matches for term: {}", term);
                if !results.is_empty() {
                    assert!(
                        results[0].score > 0.0,
                        "Expected positive score for matching term: {}, got: {}",
                        term,
                        results[0].score
                    );
                }
            } else {
                assert!(
                    results.is_empty() || results[0].score < 0.1,
                    "Expected low/no score for non-matching term: {}",
                    term
                );
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_pdf_handling() -> Result<()> {
        let (system, _temp_dir) = setup_test_system().await;

        // Create a test PDF file
        let test_dir = TempDir::new()?;
        let pdf_path = test_dir.path().join("test.pdf");

        // You might want to create a small PDF file here for testing
        // For now, we'll just test that it properly rejects non-PDF files
        fs::write(&pdf_path, b"Not a real PDF file")?;

        let result = system.index_file(pdf_path).await;
        assert!(result.is_err(), "Should fail on invalid PDF file");

        Ok(())
    }

    #[tokio::test]
    async fn test_unsupported_file_types() -> Result<()> {
        let (system, _temp_dir) = setup_test_system().await;

        // Create a test file with unsupported extension
        let test_dir = TempDir::new()?;
        let test_file = test_dir.path().join("test.xyz");
        fs::write(&test_file, "Some content")?;

        let result = system.index_file(test_file).await;
        assert!(matches!(result, Err(FileEmbeddingError::UnsupportedFileType(_))));

        Ok(())
    }

    #[tokio::test]
    async fn test_directory_indexing() -> Result<()> {
        let (system, _temp_dir) = setup_test_system().await;

        // Create a test directory structure
        let test_dir = TempDir::new()?;

        let files = [
            ("test1.txt", "Rust programming content"),
            ("test2.md", "Python scripting guide"),
            ("subfolder/test3.txt", "More Rust content"),
        ];

        println!("Creating test files...");
        for (path, content) in files {
            let full_path = test_dir.path().join(path);
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&full_path, content)?;
            println!("Created file: {} with content: {}", full_path.display(), content);
        }

        // Test directory indexing
        println!("Indexing directory...");
        system.index_directory(test_dir.path().to_path_buf()).await?;

        // Verify files are in the database with a proper count query
        #[derive(Debug, Deserialize)]
        struct CountResult {
            count: i64,
        }

        let count: Vec<CountResult> = system.db
            .query("SELECT count() as count FROM files").await?
            .take(0)?;

        println!(
            "Files in database: {}",
            count.first().map_or(0, |c| c.count)
        );

        // Print all files in database
        let all_files: Vec<FileRecord> = system.db.query("SELECT * FROM files").await?.take(0)?;
        println!("Files in database:");
        for file in &all_files {
            println!("- {} ({})", file.name, file.content_preview);
        }

        // Now test search
        println!("\nSearching for 'rust'...");
        let rust_results = system.hybrid_search("rust", 5).await?;
        println!("\nSearching for 'python'...");
        let python_results = system.hybrid_search("python", 5).await?;

        assert!(
            rust_results.len() >= 2,
            "Should find at least 2 Rust-related files. Found: {}. Results: {:?}",
            rust_results.len(),
            rust_results
        );
        assert!(
            python_results.len() >= 1,
            "Should find at least 1 Python-related file. Found: {}. Results: {:?}",
            python_results.len(),
            python_results
        );

        Ok(())
    }
    
    #[tokio::test]
    async fn test_search_quality() -> Result<()> {
        let (system, _temp_dir) = setup_test_system().await;

        println!("\n=== Starting Search Quality Test ===");

        // Create test files with more varied content
        let test_files = [
            (
                "rust_doc.txt",
                "Rust is a systems programming language focused on safety, speed, and concurrency. It prevents segfaults and ensures thread safety.",
            ),
            (
                "python_doc.txt",
                "Python is a high-level programming language known for its simplicity and readability. Great for data science.",
            ),
            (
                "mixed_doc.txt",
                "While Python is great for rapid development, Rust provides memory safety without garbage collection.",
            ),
            ("unrelated.txt", "The weather is nice today. Birds are singing in the trees."),
        ];

        // Index all files
        println!("\nIndexing test files:");
        for (name, content) in test_files {
            println!("\n--- Indexing {} ---\nContent: {}", name, content);
            let test_dir = TempDir::new()?;
            let file_path = test_dir.path().join(name);
            fs::write(&file_path, content)?;
            system.index_file(file_path).await?;
        }

        // Verify indexed content
        let all_files: Vec<FileRecord> = system.db.query("SELECT * FROM files").await?.take(0)?;
        println!("\nVerifying indexed files:");
        for file in &all_files {
            println!(
                "\nFile: {}\nPreview: {}\nEmbedding size: {}",
                file.name,
                file.content_preview,
                file.content_embedding.len()
            );
        }

        // Test different search queries with detailed output
        let search_tests = [
            ("rust safety", vec!["rust_doc.txt", "mixed_doc.txt"]),
            ("python data", vec!["python_doc.txt"]),
            ("programming language", vec!["rust_doc.txt", "python_doc.txt"]),
            ("weather", vec!["unrelated.txt"]),
        ];

        println!("\n=== Running Search Tests ===");
        for (query, expected_files) in search_tests {
            println!("\n--- Testing search for: '{}' ---", query);
            println!("Expected files: {:?}", expected_files);

            let results = system.hybrid_search(query, 5).await?;

            println!("\nResults (in order of relevance):");
            for (i, result) in results.iter().enumerate() {
                println!(
                    "{}. {} (Score: {:.4})\n   Preview: {}",
                    i + 1,
                    result.file.name,
                    result.score,
                    result.file.content_preview.chars().take(50).collect::<String>()
                );
            }

            // Verification with debug output
            let result_files: Vec<&str> = results
                .iter()
                .map(|r| r.file.name.as_str())
                .collect();

            for expected in &expected_files {
                let found = result_files.contains(expected);
                println!("Checking for {}: {}", expected, if found {
                    "FOUND ✓"
                } else {
                    "NOT FOUND ✗"
                });
                assert!(
                    found,
                    "Expected to find {} in results for query '{}', but got {:?}",
                    expected,
                    query,
                    result_files
                );
            }

            // Verify scores
            if let Some(first_score) = results.first().map(|r| r.score) {
                println!("\nScore analysis:");
                println!("Top score: {:.4}", first_score);
                println!(
                    "Score range: {:.4} to {:.4}",
                    results
                        .iter()
                        .map(|r| r.score)
                        .fold(f32::INFINITY, f32::min),
                    results
                        .iter()
                        .map(|r| r.score)
                        .fold(f32::NEG_INFINITY, f32::max)
                );
            }
        }

        println!("\n=== Search Quality Test Complete ===");
        Ok(())
    }

    #[tokio::test]
    async fn test_edge_cases() -> Result<()> {
        let (system, _temp_dir) = setup_test_system().await;

        // Create and index a test file
        let test_dir = TempDir::new()?;
        let file_path = test_dir.path().join("test.txt");
        fs::write(&file_path, "Sample content for testing")?;
        system.index_file(file_path).await?;

        // Test empty query
        let results = system.hybrid_search("", 5).await?;
        assert!(!results.is_empty(), "Should handle empty queries gracefully");

        // Test very long query
        let long_query = "a".repeat(1000);
        let results = system.hybrid_search(&long_query, 5).await?;
        assert!(results[0].score >= -1.0 && results[0].score <= 1.0, "Should handle long queries");

        // Test special characters
        let special_query = "!@#$%^&*()";
        let results = system.hybrid_search(special_query, 5).await?;
        assert!(
            results[0].score >= -1.0 && results[0].score <= 1.0,
            "Should handle special characters"
        );

        Ok(())
    }
}
