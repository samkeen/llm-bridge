//! This module defines the data models used for interacting with different LLM APIs.
//!
//! The main strategy employed to support multiple LLM APIs is to define separate response
//! structs for each API (`AnthropicResponse` and `OpenAIResponse`) and use an enum
//! (`ResponseMessage`) to represent the different response types. The `ResponseMessage` enum
//! provides a unified interface for accessing common fields and methods across different APIs.
//!
//! To add support for a new LLM API:
//! 1. Define a new response struct for the API, implementing the necessary deserialization logic.
//! 2. Add a new variant to the `ResponseMessage` enum for the new API response type.
//! 3. Update the implementation of the `ResponseMessage` methods to handle the new variant and
//!    provide the appropriate logic for accessing the fields and data.
//!
//! By following this approach, the `ResponseMessage` enum acts as a common interface for handling
//! responses from different LLM APIs, while the individual response structs encapsulate the
//! specific details of each API's response format.

use serde::{Deserialize, Serialize};

/// Represents a message in the conversation.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Represents the request body sent to the Anthropic API.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct RequestBody {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub system: String,
}




