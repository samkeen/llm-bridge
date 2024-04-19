use log::{debug, error};
use crate::error::ApiError;
use crate::models::{ChatResponse, Message, RequestBody, ResponseMessage};
use reqwest::Client;

const API_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MODEL: &str = "claude-3-haiku-20240307";
const DEFAULT_MAX_TOKENS: u32 = 100;
const DEFAULT_TEMP: f32 = 1.0;

/// Represents a builder for constructing a request to the Anthropic API.
///
/// The `RequestBuilder` allows setting various parameters for the request, such as the model,
/// messages, max tokens, temperature, and system prompt. The `send` method sends the request
/// to the API and returns the response.
pub struct RequestBuilder<'a> {
    client: &'a AnthropicClient,
    model: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    system_prompt: Option<String>,
}

impl<'a> RequestBuilder<'a> {
    /// Creates a new instance of `RequestBuilder` with the provided `AnthropicClient`.
    pub fn new(client: &'a AnthropicClient) -> Self {
        RequestBuilder {
            client,
            model: None,
            messages: None,
            max_tokens: None,
            temperature: None,
            system_prompt: None,
        }
    }

    /// Sets the model to use for the request.
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Sets the messages to include in the request.
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }

    /// Sets the maximum number of tokens to generate in the response.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the temperature value to control the randomness of the generated response.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the system prompt to provide context or instructions for the request.
    pub fn system_prompt(mut self, system_prompt: &str) -> Self {
        self.system_prompt = Some(system_prompt.to_string());
        self
    }

    /// Sends the request to the Anthropic API and returns the response.
    ///
    /// # Returns
    ///
    /// A `ResponseMessage` instance containing the API response.
    pub async fn send(self) -> Result<ResponseMessage, ApiError> {
        let model = self.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let messages = self.messages.ok_or(ApiError::MissingMessages)?;
        let max_tokens = self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let temperature = self.temperature.unwrap_or(DEFAULT_TEMP);
        let system_prompt = self.system_prompt.unwrap_or_default();

        self.client.send_message_inner(
            &model,
            messages,
            max_tokens,
            temperature,
            &system_prompt,
        ).await
    }
}

/// Wrapper around the Anthropic LLM API client.
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

    /// Returns a new `RequestBuilder` instance for constructing a request.
    pub fn request(&self) -> RequestBuilder {
        RequestBuilder::new(self)
    }

    /// Sends a single message to the Anthropic API and retrieves the response.
    ///
    /// # Arguments
    ///
    /// * `model` - The name of the model to use for generating the response.
    /// * `messages` - The list of messages to send to the API.
    /// * `max_tokens` - The maximum number of tokens to generate in the response.
    /// * `temperature` - The temperature value to control the randomness of the generated response.
    /// * `system_prompt` - The system prompt to provide context or instructions for the request.
    ///
    /// # Returns
    ///
    /// A `ResponseMessage` instance containing the API response.
    async fn send_message_inner(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
        system_prompt: &str,
    ) -> Result<ResponseMessage, ApiError> {
        let body = RequestBody {
            model: model.to_string(),
            messages,
            max_tokens,
            temperature,
            system: system_prompt.to_string(),
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
    pub fn chat(&self, model: &str, max_tokens: u32, temperature: f32, system_prompt: Option<String>) -> ChatSession {
        ChatSession {
            client: self,
            model: model.to_string(),
            messages: Vec::new(),
            max_tokens,
            temperature,
            system_prompt,
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
    system_prompt: Option<String>,
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

        let response = self.client
            .request()
            .model(&self.model)
            .messages(self.messages.clone())
            .max_tokens(self.max_tokens)
            .temperature(self.temperature)
            .system_prompt(self.system_prompt.as_deref().unwrap_or(""))
            .send()
            .await?;

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