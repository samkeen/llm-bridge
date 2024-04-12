use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnthropicError {
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),
    // Add more error variants as needed
}