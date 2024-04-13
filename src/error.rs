use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnthropicError {
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("JSON Error {0}")] // Add this variant
    JsonError(#[from] serde_json::Error),
    // Add more error variants as needed
}