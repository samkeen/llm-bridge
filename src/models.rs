use fmt::Display;
use std::fmt;
use serde::{Deserialize, Serialize};
use crate::client::ChatSession;
use crate::error::ApiError;

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
    /// A system prompt is a way of providing context and instructions to Claude, such as 
    /// specifying a particular goal or role. 
    /// https://docs.anthropic.com/claude/docs/system-prompts
    pub system: String,
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

impl ResponseMessage {
    /// Returns the text content of the first content block in the response message.
    ///
    /// This method retrieves the first content block from the `content` vector of the `ResponseMessage`
    /// and returns its text content as a `String`. If the `content` vector is empty, an empty string
    /// is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_api_adapter::models::ResponseMessage;
    ///
    /// let response_message = ResponseMessage {
    ///     id: "123".to_string(),
    ///     role: "assistant".to_string(),
    ///     content: vec![
    ///         ContentBlock {
    ///             text: "Hello, how can I assist you today?".to_string(),
    ///             block_type: "text".to_string(),
    ///         },
    ///         ContentBlock {
    ///             text: "Let me know if you have any questions!".to_string(),
    ///             block_type: "text".to_string(),
    ///         },
    ///     ],
    /// };
    ///
    /// let first_message = response_message.first_message();
    /// assert_eq!(first_message, "Hello, how can I assist you today?");
    /// ```
    ///
    /// # Returns
    ///
    /// A `String` containing the text content of the first content block in the response message.
    /// If the `content` vector is empty, an empty string is returned.
    pub fn first_message(&self) -> String {
        let content = self.content.first();
        match content {
            None => {
                "".to_string()
            }
            Some(content) => {
                content.text.to_string()
            }
        }
    }
}

/// Implement Display trait for ResponseMessage
impl Display for ResponseMessage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ResponseMessage {{ id: {}, role: {}, content: {:?} }}", self.id, self.role, self.content)
    }
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
    pub async fn add(self, message: &str) -> Result<ChatResponse<'a>, ApiError> {
        self.session.send(message).await
    }
}
