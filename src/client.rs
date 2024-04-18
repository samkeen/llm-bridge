use log::{debug, error};
use crate::error::ApiError;
use crate::models::{ChatResponse, Message, RequestBody, ResponseMessage};
use reqwest::Client;

const API_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

/// Wrapper around the Anthropic LLM API client
pub struct AnthropicClient {
    api_key: String,
    client: Client,
}

impl AnthropicClient {
    /// Creates a new instance of `AnthropicClient` with the provided API key.
    pub fn new(api_key: String) -> Self {
        let client = Client::new();
        AnthropicClient { api_key, client }
    }

    /// Sends a single message to the Anthropic API and retrieves the response.
    ///
    /// # Arguments
    ///
    /// * `model` - The name of the model to use for generating the response.
    /// * `messages` - The list of messages to send to the API.
    /// * `max_tokens` - The maximum number of tokens to generate in the response.
    /// * `temperature` - The temperature value to control the randomness of the generated response.
    ///
    /// # Returns
    ///
    /// A `ResponseMessage` instance containing the API response.
    pub async fn send_message(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ResponseMessage, ApiError> {
        let body = RequestBody {
            model: model.to_string(),
            messages,
            max_tokens,
            temperature,
        };

        let response = self
            .client
            .post(API_ENDPOINT)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;
        let resp_status = response.status();
        let resp_text = response.text().await.unwrap_or("".into());
        if resp_status.is_client_error() {
            // Handle 4xx error responses
            error!("Client error [{}]: {}", resp_status, resp_text);
            return Err(ApiError::ClientError(
                format!("Status: {} - Error: {}", resp_status, resp_text)));
        } else if resp_status.is_server_error() {
            // Handle 5xx error responses
            error!("Server error [{}]: {}", resp_status, resp_text);
            return Err(ApiError::ServerError(
                format!("Status: {} - Error: {}", resp_status, resp_text)));
        }
        debug!("LLM call response: status[{}]\n{}", resp_status, resp_text);
        let response_message = serde_json::from_str(&resp_text)?;

        Ok(response_message)
    }

    /// Creates a new `ChatSession` with the specified model, max tokens, and temperature.
    ///
    /// # Arguments
    ///
    /// * `model` - The name of the model to use for the chat session.
    /// * `max_tokens` - The maximum number of tokens to generate in each response.
    /// * `temperature` - The temperature value to control the randomness of the generated responses.
    ///
    /// # Returns
    ///
    /// A new `ChatSession` instance.
    pub fn chat(&self, model: &str, max_tokens: u32, temperature: f32) -> ChatSession {
        ChatSession {
            client: self,
            model: model.to_string(),
            messages: Vec::new(),
            max_tokens,
            temperature,
        }
    }
}

/// Represents an ongoing chat session with the Anthropic API.
///
/// The lifetime parameter `'a` indicates that the `ChatSession` borrows data from an `AnthropicClient`
/// instance and can only live as long as the `AnthropicClient` instance.
pub struct ChatSession<'a> {
    client: &'a AnthropicClient,
    model: String,
    pub(crate) messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

impl<'a> ChatSession<'a> {
    /// Sends a user message to the chat session and retrieves the response from the API.
    ///
    /// # Arguments
    ///
    /// * `message` - The user message to send.
    ///
    /// # Returns
    ///
    /// A `ChatResponse` instance containing the last response and the updated chat session.
    pub async fn send(mut self, message: &str) -> Result<ChatResponse<'a>, ApiError> {
        self.messages.push(Message {
            role: "user".to_string(),
            content: message.to_string(),
        });

        let response = self.client.send_message(
            &self.model,
            self.messages.clone(),
            self.max_tokens,
            self.temperature,
        ).await?;

        let content = response.content.into_iter()
            .map(|block| block.text)
            .collect::<String>();

        self.messages.push(Message {
            role: response.role,
            content: content.clone(),
        });

        Ok(ChatResponse {
            session: self,
            last_response: content,
        })
    }
}