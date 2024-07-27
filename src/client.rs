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
use crate::request::{Message, RequestBody};
use reqwest::Client;
use serde_json::{json, Number};
use crate::response::{OpenAIResponse, ResponseMessage};
use crate::tool::Tool;

const API_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-3-haiku-20240307";

const DEFAULT_OPENAI_MODEL: &str = "gpt-4o";
const DEFAULT_MAX_TOKENS: u32 = 100;
const DEFAULT_TEMP: f64 = 0.0;

#[derive(Debug, Clone)]
/// Supported LLMs
pub enum ClientLlm {
    Anthropic,
    OpenAI,
}

#[async_trait::async_trait]
pub trait LlmClientTrait: Send + Sync {
    async fn send_message(&self, request_body: serde_json::Value) -> Result<ResponseMessage, ApiError>;
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
    temperature: Option<f64>,
    system_prompt: Option<String>,
    tools: Option<Vec<Tool>>
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
            tools: None,
        }
    }

    pub fn add_tool(mut self, tool: Tool) -> Self {
        if let Some(mut tools) = self.tools {
            tools.push(tool);
            self.tools = Some(tools);
        } else {
            self.tools = Some(vec![tool]);
        }
        self
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
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the system prompt to provide context and instructions to the model.
    pub fn system_prompt(mut self, system_prompt: &str) -> Self {
        self.system_prompt = Some(system_prompt.to_string());
        self
    }

    pub fn render_request(&self) -> Result<serde_json::Value, ApiError> {
        let model = self.model.clone().unwrap_or_else(|| {
            match self.client.client_type() {
                ClientLlm::Anthropic => DEFAULT_ANTHROPIC_MODEL.to_string(),
                ClientLlm::OpenAI => DEFAULT_OPENAI_MODEL.to_string(),
                // Add more cases for other LLM APIs as needed
            }
        });
        let messages = self.messages.clone().ok_or(ApiError::MissingMessages)?;
        let max_tokens = self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let temperature = self.temperature.unwrap_or(DEFAULT_TEMP);
        let temperature_number = Number::from_f64(temperature)
            .ok_or_else(|| ApiError::InvalidUsage(format!("Invalid temperature value: {}", temperature)))?;
        let system_prompt = self.system_prompt.clone().unwrap_or_default();

        match self.client.client_type() {
            ClientLlm::Anthropic => {
                let mut request = json!({
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature_number,
                    "system": system_prompt,
                });

                if let Some(tools) = &self.tools {
                    let anthropic_tools: Vec<serde_json::Value> = tools.iter()
                        .map(|tool| tool.to_anthropic_format())
                        .collect();
                    request["tools"] = json!(anthropic_tools);
                }

                Ok(request)
            },
            ClientLlm::OpenAI => {
                let mut request = json!({
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature_number,
                });

                if !system_prompt.is_empty() {
                    request["messages"].as_array_mut().unwrap().push(json!({
                        "role": "system",
                        "content": system_prompt
                    }));
                }

                if let Some(tools) = &self.tools {
                    let openai_tools: Vec<serde_json::Value> = tools.iter()
                        .map(|tool| tool.to_openai_format())
                        .collect();
                    request["tools"] = json!(openai_tools);
                }

                Ok(request)
            },
        }
    }


    pub async fn send(self) -> Result<ResponseMessage, ApiError> {
        let request_body = self.render_request()?;
        self.client.send_message(request_body).await
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
    async fn send_message(&self, request_body: serde_json::Value) -> Result<ResponseMessage, ApiError> {
        let response = self.client
            .post(API_ENDPOINT)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&request_body)
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
    async fn send_message(&self, request_body: serde_json::Value) -> Result<ResponseMessage, ApiError> {
        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let resp_status = response.status();
        let resp_text = response.text().await.unwrap_or("".into());
        if resp_status.is_client_error() {
            return Err(ApiError::ClientError(format!("Status: {} - Error: {}", resp_status, resp_text)));
        } else if resp_status.is_server_error() {
            return Err(ApiError::ServerError(format!("Status: {} - Error: {}", resp_status, resp_text)));
        }

        let openai_response: OpenAIResponse = serde_json::from_str(&resp_text)?;
        Ok(ResponseMessage::OpenAI(openai_response))
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

#[cfg(test)]
mod tests {
    use dotenv::dotenv;
    use super::*;
    use crate::tool::Tool;

    struct MockClient {
        client_type: ClientLlm,
    }

    #[async_trait::async_trait]
    impl LlmClientTrait for MockClient {
        async fn send_message(&self, _request_body: serde_json::Value) -> Result<ResponseMessage, ApiError> {
            unimplemented!()
        }

        fn client_type(&self) -> ClientLlm {
            self.client_type.clone()
        }
    }

    #[test]
    fn test_anthropic_default_request() {
        let client = MockClient { client_type: ClientLlm::Anthropic };
        let builder = RequestBuilder::new(&client)
            .user_message("Hello, Claude!");

        let request = builder.render_request().unwrap();

        assert_eq!(request["model"], DEFAULT_ANTHROPIC_MODEL);
        assert_eq!(request["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(request["temperature"], DEFAULT_TEMP);
        assert_eq!(request["system"], "");
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "Hello, Claude!");
    }

    #[test]
    fn test_openai_default_request() {
        let client = MockClient { client_type: ClientLlm::OpenAI };
        let builder = RequestBuilder::new(&client)
            .user_message("Hello, GPT!");

        let request = builder.render_request().unwrap();

        assert_eq!(request["model"], DEFAULT_OPENAI_MODEL);
        assert_eq!(request["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(request["temperature"], DEFAULT_TEMP);
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "Hello, GPT!");
    }

    #[test]
    fn test_custom_model_and_parameters() {
        let client = MockClient { client_type: ClientLlm::Anthropic };
        let builder = RequestBuilder::new(&client)
            .model("custom-model")
            .max_tokens(500)
            .temperature(0.8)
            .system_prompt("You are a helpful assistant.")
            .user_message("Tell me a joke.");

        let request = builder.render_request().unwrap();

        assert_eq!(request["model"], "custom-model");
        assert_eq!(request["max_tokens"], 500);

        // Check for exact temperature value
        assert_eq!(request["temperature"], json!(0.8));

        assert_eq!(request["system"], "You are a helpful assistant.");
        assert_eq!(request["messages"][0]["content"], "Tell me a joke.");
    }

    #[test]
    fn test_multiple_messages() {
        let client = MockClient { client_type: ClientLlm::OpenAI };
        let builder = RequestBuilder::new(&client)
            .user_message("Hello!")
            .user_message("How are you?");

        let request = builder.render_request().unwrap();

        assert_eq!(request["messages"].as_array().unwrap().len(), 2);
        assert_eq!(request["messages"][0]["content"], "Hello!");
        assert_eq!(request["messages"][1]["content"], "How are you?");
    }

    #[test]
    fn test_missing_messages() {
        let client = MockClient { client_type: ClientLlm::Anthropic };
        let builder = RequestBuilder::new(&client);

        let result = builder.render_request();

        assert!(matches!(result, Err(ApiError::MissingMessages)));
    }

    #[test]
    fn test_openai_system_prompt() {
        let client = MockClient { client_type: ClientLlm::OpenAI };
        let builder = RequestBuilder::new(&client)
            .system_prompt("You are a helpful assistant.")
            .user_message("Hello!");

        let request = builder.render_request().unwrap();

        assert_eq!(request["messages"].as_array().unwrap().len(), 2);
        assert_eq!(request["messages"][1]["role"], "system");
        assert_eq!(request["messages"][1]["content"], "You are a helpful assistant.");
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "Hello!");
    }

    #[test]
    fn test_default_temperature() {
        let client = MockClient { client_type: ClientLlm::Anthropic };
        let builder = RequestBuilder::new(&client)
            .user_message("Test message");

        let request = builder.render_request().unwrap();

        assert_eq!(request["temperature"], json!(DEFAULT_TEMP));
    }

    #[test]
    fn test_custom_temperature() {
        let client = MockClient { client_type: ClientLlm::Anthropic };
        let custom_temp = 0.7;
        let builder = RequestBuilder::new(&client)
            .temperature(custom_temp)
            .user_message("Test message");

        let request = builder.render_request().unwrap();

        assert_eq!(request["temperature"], json!(custom_temp));
    }

    #[test]
    fn test_temperature_precision() {
        let client = MockClient { client_type: ClientLlm::Anthropic };
        let precise_temp = 0.12345;
        let builder = RequestBuilder::new(&client)
            .temperature(precise_temp)
            .user_message("Test message");

        let request = builder.render_request().unwrap();

        assert_eq!(request["temperature"], json!(precise_temp));
    }

    #[test]
    fn test_invalid_temperature() {
        use std::f64::{INFINITY, NEG_INFINITY};

        let client = MockClient { client_type: ClientLlm::Anthropic };

        for &invalid_temp in &[INFINITY, NEG_INFINITY, f64::NAN] {
            let builder = RequestBuilder::new(&client)
                .temperature(invalid_temp)
                .user_message("Test message");

            let result = builder.render_request();
            assert!(matches!(result, Err(ApiError::InvalidUsage(_))));
        }
    }
    
    fn get_weather_tool() -> Tool {
        Tool::builder()
            .name("get_weather")
            .description("Get the current weather in a given location")
            .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
            .add_enum_parameter("unit", "The unit of temperature, either 'celsius' or 'fahrenheit'", false, vec!["celsius".to_string(), "fahrenheit".to_string()])
            .build()
            .expect("Failed to build tool")
    }

    #[test]
    fn test_tool_use_anthropic() {
        dotenv().ok();
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY must be set.");
        let client_type = ClientLlm::Anthropic;
        let mut client = LlmClient::new(client_type, api_key);

        let tool = get_weather_tool();

        let request = client
            .request()
            .add_tool(tool)
            .model("claude-3-haiku-20240307")
            .user_message("What is the current weather in San Francisco, California")
            .max_tokens(100)
            .temperature(1.0)
            .system_prompt("You are a haiku assistant.")
            .render_request()
            .expect("Failed to render request");

        // Check if the tools field is present and correctly formatted
        assert!(request.get("tools").is_some(), "Tools field is missing");
        let tools = request["tools"].as_array().expect("Tools should be an array");
        assert_eq!(tools.len(), 1, "There should be one tool");

        let tool = &tools[0];
        assert_eq!(tool["name"], "get_weather", "Tool name should be 'get_weather'");
        assert!(tool["input_schema"].is_object(), "Tool should have an input schema");

        let input_schema = &tool["input_schema"];
        assert_eq!(input_schema["type"], "object", "Input schema type should be 'object'");

        let properties = input_schema["properties"].as_object().expect("Properties should be an object");
        assert!(properties.contains_key("location"), "Location parameter should be present");
        assert!(properties.contains_key("unit"), "Unit parameter should be present");

    }

    #[test]
    fn test_function_calling_openai() {
        dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY must be set.");
        let client_type = ClientLlm::OpenAI;
        let mut client = LlmClient::new(client_type, api_key);

        let tool = get_weather_tool();

        let request = client
            .request()
            .add_tool(tool)
            .model("gpt-4o")
            .user_message("What is the current weather in San Francisco, California")
            .max_tokens(100)
            .temperature(1.0)
            .system_prompt("You are a weather assistant.")
            .render_request()
            .expect("Failed to render request");

        // Check if the functions field is present and correctly formatted
        assert!(request.get("tools").is_some(), "Tools field is missing");
        let tools = request["tools"].as_array().expect("Tools should be an array");
        assert_eq!(tools.len(), 1, "There should be one tool");

        let function = &tools[0];
        assert_eq!(function["type"], "function", "Tool type should be 'function'");

        let function_details = &function["function"];
        assert_eq!(function_details["name"], "get_weather", "Function name should be 'get_weather'");
        assert_eq!(function_details["description"], "Get the current weather in a given location", "Function description should match");

        let parameters = &function_details["parameters"];
        assert_eq!(parameters["type"], "object", "Parameters type should be 'object'");

        let properties = parameters["properties"].as_object().expect("Properties should be an object");
        assert!(properties.contains_key("location"), "Location parameter should be present");
        assert!(properties.contains_key("unit"), "Unit parameter should be present");

        let location = &properties["location"];
        assert_eq!(location["type"], "string", "Location type should be 'string'");

        let unit = &properties["unit"];
        assert_eq!(unit["type"], "string", "Unit type should be 'string'");
        assert!(unit.get("enum").is_some(), "Unit should have enum values");

        let required = parameters["required"].as_array().expect("Required should be an array");
        assert!(required.contains(&json!("location")), "Location should be a required parameter");

        // Check other request parameters
        assert_eq!(request["model"], "gpt-4o", "Model should be set correctly");
        assert_eq!(request["max_tokens"], 100, "Max tokens should be set correctly");
        assert_eq!(request["temperature"], 1.0, "Temperature should be set correctly");

        // Check that the system message is included in the messages array
        let messages = request["messages"].as_array().expect("Messages should be an array");
        assert!(messages.iter().any(|msg| msg["role"] == "system" && msg["content"] == "You are a weather assistant."),
                "System message should be included in the messages array");
    }
}