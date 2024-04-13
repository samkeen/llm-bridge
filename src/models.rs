use serde::{Deserialize, Serialize};
use crate::client::ChatSession;
use crate::error::AnthropicError;

/// Represents a message in the conversation.
///
/// A `Message` struct contains the role of the sender (either "user" or "assistant")
/// and the content of the message.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    /// The role of the sender, either "user" or "assistant".
    pub role: String,
    /// The content of the message.
    pub content: String,
}

/// Represents the request body sent to the Anthropic API.
///
/// The `RequestBody` struct contains the model name, the list of messages,
/// the maximum number of tokens to generate, and the temperature value.
#[derive(Serialize, Deserialize, Debug)]
pub struct RequestBody {
    /// The name of the model to use for generating the response.
    pub model: String,
    /// The list of messages in the conversation.
    pub messages: Vec<Message>,
    /// The maximum number of tokens to generate in the response.
    pub max_tokens: u32,
    /// The temperature value to control the randomness of the generated response.
    pub temperature: f32,
}

/// Represents a block of content in the API response.
///
/// A `ContentBlock` struct contains the text content and the type of the block.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContentBlock {
    /// The text content of the block.
    pub text: String,
    /// The type of the content block.
    ///
    /// The `#[serde(rename = "type")]` attribute is used to map the `type` field
    /// in the JSON response to the `block_type` field in the struct.
    #[serde(rename = "type")]
    pub block_type: String,
}

/// Represents the response message received from the Anthropic API.
///
/// The `ResponseMessage` struct contains the ID of the response, the role of the sender,
/// and the list of content blocks in the response.
#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseMessage {
    /// The ID of the response message.
    pub id: String,
    /// The role of the sender, either "user" or "assistant".
    pub role: String,
    /// The list of content blocks in the response.
    pub content: Vec<ContentBlock>,
}

/// Represents the response from a chat session.
///
/// The lifetime parameter `'a` indicates that the `ChatResponse` borrows data from a `ChatSession`
/// instance and can only live as long as the `ChatSession` instance.
pub struct ChatResponse<'a> {
    pub(crate) session: ChatSession<'a>,
    pub(crate) last_response: String,
}

impl<'a> ChatResponse<'a> {
    /// Returns the last response from the chat session.
    pub fn last_response(&self) -> &str {
        &self.last_response
    }

    /// Returns the entire conversation history of the chat session.
    pub fn dialog(&self) -> &[Message] {
        &self.session.messages
    }

    /// Sends a new user message to continue the conversation and returns the updated `ChatResponse`.
    ///
    /// # Arguments
    ///
    /// * `message` - The user message to send.
    ///
    /// # Returns
    ///
    /// An updated `ChatResponse` instance containing the last response and the updated chat session.
    pub async fn add(self, message: &str) -> Result<ChatResponse<'a>, AnthropicError> {
        self.session.send(message).await
    }
}
