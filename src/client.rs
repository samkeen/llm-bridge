//! This module provides a client for interacting with different LLM APIs.
//!
//! The `LlmClient` struct is the main entry point for making requests to LLM APIs.
//! It uses a `RequestBuilder` to construct the request parameters and sends the request
//! using the appropriate client implementation based on the selected `ClientLlm` enum variant.
//!
//! The `LlmClientTrait` defines the common interface for sending messages to LLM APIs,
//! and the `AnthropicClient` and `OpenAIClient` structs implement this trait for their respective APIs.

use log::{debug, error};
use crate::error::ApiError;
use crate::models::{Message, RequestBody, ResponseMessage};
use reqwest::Client;

const API_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-3-haiku-20240307";

const DEFAULT_OPENAI_MODEL: &str = "gpt-4o";
const DEFAULT_MAX_TOKENS: u32 = 100;
const DEFAULT_TEMP: f32 = 0.0;

/// Supported LLMs
pub enum ClientLlm {
    Anthropic,
    OpenAI,
}

/// Trait defining the common interface for LLM clients.
#[async_trait::async_trait]
pub trait LlmClientTrait: Send + Sync {
    /// Sends a message to the LLM API and returns the response.
    ///
    /// # Arguments
    ///
    /// * `model` - The name of the model to use for generating the response.
    /// * `messages` - The list of messages in the conversation.
    /// * `max_tokens` - The maximum number of tokens to generate in the response.
    /// * `temperature` - The temperature value to control the randomness of the generated response.
    /// * `system_prompt` - The system prompt to provide context and instructions to the model.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `ResponseMessage` on success, or an `ApiError` on failure.
    async fn send_message(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
        system_prompt: &str,
    ) -> Result<ResponseMessage, ApiError>;

    fn client_type(&self) -> ClientLlm;
}

/// Represents a builder for constructing a request to the Anthropic API.
///
/// The `RequestBuilder` allows setting various parameters for the request, such as the model,
/// messages, max tokens, temperature, and system prompt. The `send` method sends the request
/// to the API and returns the response.
pub struct RequestBuilder<'a> {
    client: &'a (dyn LlmClientTrait + Send + Sync),
    model: Option<String>,
    messages: Option<Vec<Message>>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    system_prompt: Option<String>,
}

impl<'a> RequestBuilder<'a> {
    pub fn new(client: &'a (dyn LlmClientTrait + Send + Sync)) -> Self {
        RequestBuilder {
            client,
            model: None,
            messages: None,
            max_tokens: None,
            temperature: None,
            system_prompt: None,
        }
    }

    /// Sets the model to use for generating the response.
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Adds a user message to the conversation.
    pub fn user_message(mut self, message: &str) -> Self {
        if let Some(mut messages) = self.messages {
            messages.push(Message {
                role: "user".to_string(),
                content: message.to_string(),
            });
            self.messages = Some(messages);
        } else {
            self.messages = Some(vec![Message {
                role: "user".to_string(),
                content: message.to_string(),
            }]);
        }
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

    /// Sets the system prompt to provide context and instructions to the model.
    pub fn system_prompt(mut self, system_prompt: &str) -> Self {
        self.system_prompt = Some(system_prompt.to_string());
        self
    }

    /// Sends the request to the LLM API and returns the response.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use llm_bridge::client::{LlmClient, ClientLlm};
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let api_key = "your_api_key".to_string();
    ///     let client_type = ClientLlm::Anthropic;
    ///     let mut client = LlmClient::new(client_type, api_key);
    ///
    ///     let response = client
    ///         .request()
    ///         .model("claude-3-haiku-20240307")
    ///         .user_message("Hello, Claude!")
    ///         .max_tokens(100)
    ///         .temperature(1.0)
    ///         .system_prompt("You are a haiku assistant.")
    ///         .send()
    ///         .await
    ///         .expect("Failed to send message");
    ///
    ///     println!("Response: {}", response.first_message());
    /// }
    /// ```
    pub async fn send(self) -> Result<ResponseMessage, ApiError> {
        let model = self.model.unwrap_or_else(|| {
            match self.client.client_type() {
                ClientLlm::Anthropic => DEFAULT_ANTHROPIC_MODEL.to_string(),
                ClientLlm::OpenAI => DEFAULT_OPENAI_MODEL.to_string(),
                // Add more cases for other LLM APIs as needed
            }
        });
        let messages = self.messages.ok_or(ApiError::MissingMessages)?;
        let max_tokens = self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let temperature = self.temperature.unwrap_or(DEFAULT_TEMP);
        let system_prompt = self.system_prompt.unwrap_or_default();

        self.client.send_message(
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
    pub fn new(api_key: String) -> Self {
        let client = Client::new();
        AnthropicClient { api_key, client }
    }
}

#[async_trait::async_trait]
impl LlmClientTrait for AnthropicClient {
    async fn send_message(
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
        // https://docs.anthropic.com/en/api/messages
        //
        // Request JSON
        // {
        //     "model": "claude-3-opus-20240229",
        //     "max_tokens": 1024,
        //     "messages": [
        //         {"role": "user", "content": "Hello, world"}
        //     ]
        // }
        //
        // Response JSON
        // {
        //   "content": [
        //     {
        //       "text": "Hi! My name is Claude.",
        //       "type": "text"
        //     }
        //   ],
        //   "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        //   "model": "claude-3-opus-20240229",
        //   "role": "assistant",
        //   "stop_reason": "end_turn",
        //   "stop_sequence": null,
        //   "type": "message",
        //   "usage": {
        //     "input_tokens": 10,
        //     "output_tokens": 25
        //   }
        // }
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
            error!("Client error [{}]: {}", resp_status, resp_text);
            return Err(ApiError::ClientError(
                format!("Status: {} - Error: {}", resp_status, resp_text)));
        } else if resp_status.is_server_error() {
            error!("Server error [{}]: {}", resp_status, resp_text);
            return Err(ApiError::ServerError(
                format!("Status: {} - Error: {}", resp_status, resp_text)));
        }
        debug!("LLM call response: status[{}]\n{}", resp_status, resp_text);
        let response_message = serde_json::from_str(&resp_text)?;

        Ok(response_message)
    }

    fn client_type(&self) -> ClientLlm {
        ClientLlm::Anthropic
    }
}

/// Wrapper around the OpenAI LLM API client.
pub struct OpenAIClient {
    api_key: String,
    client: Client,
}

impl OpenAIClient {
    pub fn new(api_key: String) -> Self {
        let client = Client::new();
        OpenAIClient { api_key, client }
    }
}

#[async_trait::async_trait]
impl LlmClientTrait for OpenAIClient {
    async fn send_message(
        &self,
        model: &str,
        mut messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
        system_prompt: &str,
    ) -> Result<ResponseMessage, ApiError> {
        // OpenAI places the System prompt in the messages list
        if !system_prompt.is_empty() {
            messages.push(Message { role: "system".to_string(), content: system_prompt.to_string() });
        }
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });
        // https://platform.openai.com/docs/api-reference/making-requests
        //
        // Request JSON
        // {
        //   "model": "gpt-3.5-turbo",
        //   "messages": [{"role": "user", "content": "Say this is a test!"}],
        //   "temperature": 0.7
        // }
        //
        // Response JSON
        // {
        //     "id": "chatcmpl-abc123",
        //     "object": "chat.completion",
        //     "created": 1677858242,
        //     "model": "gpt-3.5-turbo-0613",
        //     "usage": {
        //         "prompt_tokens": 13,
        //         "completion_tokens": 7,
        //         "total_tokens": 20
        //     },
        //     "choices": [
        //         {
        //             "message": {
        //                 "role": "assistant",
        //                 "content": "\n\nThis is a test!"
        //             },
        //             "logprobs": null,
        //             "finish_reason": "stop",
        //             "index": 0
        //         }
        //     ]
        // }
        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let resp_status = response.status();
        let resp_text = response.text().await.unwrap_or("".into());
        if resp_status.is_client_error() {
            return Err(ApiError::ClientError(format!("Status: {} - Error: {}", resp_status, resp_text)));
        } else if resp_status.is_server_error() {
            return Err(ApiError::ServerError(format!("Status: {} - Error: {}", resp_status, resp_text)));
        }

        let response_message: ResponseMessage = serde_json::from_str(&resp_text)?;
        Ok(response_message)
    }

    fn client_type(&self) -> ClientLlm {
        ClientLlm::OpenAI
    }
}

/// The main client for interacting with LLM APIs.
///
/// The `LlmClient` struct provides a convenient way to make requests to LLM APIs using the
/// `RequestBuilder`. It internally uses the appropriate client implementation based on the
/// selected `ClientLlm` enum variant.
pub struct LlmClient {
    client: Box<dyn LlmClientTrait + Send + Sync>,
}

impl LlmClient {
    /// Creates a new `LlmClient` instance with the specified `ClientLlm` variant and API key.
    pub fn new(client_type: ClientLlm, api_key: String) -> Self {
        let client: Box<dyn LlmClientTrait + Send + Sync> = match client_type {
            ClientLlm::Anthropic => Box::new(AnthropicClient::new(api_key)),
            ClientLlm::OpenAI => Box::new(OpenAIClient::new(api_key)),
        };
        LlmClient { client }
    }

    /// Creates a new `RequestBuilder` for constructing a request to the LLM API.
    pub fn request(&mut self) -> RequestBuilder {
        RequestBuilder::new(self.client.as_ref())
    }
}