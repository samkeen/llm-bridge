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
use std::fmt;

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

/// Represents a block of content in the API response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnthropicContentBlock {
    pub text: String,
    #[serde(rename = "type")]
    pub block_type: String,
}

/// Tokens represent the underlying cost to llm systems.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct CommonUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct AnthropicResponse {
    pub id: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct OpenAIResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub usage: OpenAIUsage,
    pub choices: Vec<OpenAIChoice>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub(crate) struct OpenAIUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct OpenAIChoice {
    pub message: OpenAIMessage,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct OpenAIMessage {
    pub role: String,
    pub content: String,
}

/// Represents the response message received from an LLM API.
///
/// The `ResponseMessage` enum encapsulates the different response types from various LLM APIs,
/// providing a unified interface for accessing common fields and methods.
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ResponseMessage {
    Anthropic(AnthropicResponse),
    OpenAI(OpenAIResponse),
}

impl ResponseMessage {
    /// Returns the text content of the first message in the response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::models::ResponseMessage;
    /// # let response = ResponseMessage::Anthropic(/* ... */);
    /// let first_message = response.first_message();
    /// println!("First message: {}", first_message);
    /// ```
    pub fn first_message(&self) -> String {
        match self {
            ResponseMessage::Anthropic(response) => {
                if let Some(content) = response.content.first() {
                    content.text.clone()
                } else {
                    String::new()
                }
            }
            ResponseMessage::OpenAI(response) => {
                if let Some(choice) = response.choices.first() {
                    choice.message.content.clone()
                } else {
                    String::new()
                }
            }
        }
    }

    /// Returns the role of the sender in the response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::models::ResponseMessage;
    /// # let response = ResponseMessage::Anthropic(AnthropicResponse { });
    /// let role = response.role();
    /// println!("Role: {}", role);
    /// ```
    pub fn role(&self) -> &str {
        match self {
            ResponseMessage::Anthropic(response) => &response.role,
            ResponseMessage::OpenAI(response) => {
                if let Some(choice) = response.choices.first() {
                    &choice.message.role
                } else {
                    ""
                }
            }
        }
    }

    /// Returns the name of the model used for generating the response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::models::ResponseMessage;
    /// # let response = ResponseMessage::Anthropic(/* ... */);
    /// let model = response.model();
    /// println!("Model: {}", model);
    /// ```
    pub fn model(&self) -> &str {
        match self {
            ResponseMessage::Anthropic(response) => &response.model,
            ResponseMessage::OpenAI(response) => &response.model,
        }
    }

    /// Returns the stop reason for the generated response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::models::ResponseMessage;
    /// # let response = ResponseMessage::Anthropic(/* ... */);
    /// let stop_reason = response.stop_reason();
    /// println!("Stop reason: {}", stop_reason);
    /// ```
    pub fn stop_reason(&self) -> &str {
        match self {
            ResponseMessage::Anthropic(response) => &response.stop_reason,
            ResponseMessage::OpenAI(response) => {
                if let Some(choice) = response.choices.first() {
                    &choice.finish_reason
                } else {
                    ""
                }
            }
        }
    }

    /// Returns the usage information for the generated response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::models::ResponseMessage;
    /// # let response = ResponseMessage::Anthropic(/* ... */);
    /// let usage = response.usage();
    /// println!("Input tokens: {}", usage.input_tokens);
    /// println!("Output tokens: {}", usage.output_tokens);
    /// ```
    pub fn usage(&self) -> CommonUsage {
        match self {
            ResponseMessage::Anthropic(response) => CommonUsage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
            },
            ResponseMessage::OpenAI(response) => CommonUsage {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
            },
        }
    }
}

impl fmt::Display for ResponseMessage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ResponseMessage::Anthropic(response) => {
                write!(
                    f,
                    "ResponseMessage {{ id: {}, role: {}, content: {:?} }}",
                    response.id, response.role, response.content
                )
            }
            ResponseMessage::OpenAI(response) => {
                write!(
                    f,
                    "ResponseMessage {{ id: {}, object: {}, model: {}, choices: {:?} }}",
                    response.id, response.object, response.model, response.choices
                )
            }
        }
    }
}