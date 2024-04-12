use crate::error::AnthropicError;
use crate::models::{Message, RequestBody, ResponseMessage};
use reqwest::{Client};

const API_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
// Claude 3 Opus	claude-3-opus-20240229
// Claude 3 Sonnet	claude-3-sonnet-20240229
// Claude 3 Haiku	claude-3-haiku-20240307

pub struct AnthropicClient {
    api_key: String,
    client: Client,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Self {
        let client = Client::new();
        AnthropicClient { api_key, client }
    }

    pub async fn send_message(
        &self,
        model: &str,
        messages: Vec<Message>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ResponseMessage, AnthropicError> {
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

        let response_message = response.json::<ResponseMessage>().await?;
        Ok(response_message)
    }
}