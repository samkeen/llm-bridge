use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Client error returned from API: {0}")]
    ClientError(String),

    #[error("Server error returned from API: {0}")]
    ServerError(String),

    #[error("Response parse error: {0}")]
    ResponseParseError(#[from] serde_json::Error),

    #[error("Missing required 'messages' parameter")]
    MissingMessages,
    
    #[error("Invalid API Usage: {0}")]
    InvalidUsage(String),
}