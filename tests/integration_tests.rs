use dotenv::dotenv;
use llm_bridge::client::{ClientLlm, LlmClient};
use llm_bridge::error::ApiError;
use pretty_assertions::{assert_eq};


#[tokio::test]
async fn test_send_message_anthropic() {
    dotenv().ok();
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set.");
    let client_type = ClientLlm::Anthropic;
    let mut client = LlmClient::new(client_type, api_key);

    let response = client
        .request()
        .model("claude-3-haiku-20240307")
        .user_message("Hello, Claude!")
        .max_tokens(100)
        .temperature(1.0)
        .system_prompt("You are a haiku assistant.") // optional
        .send()
        .await
        .expect("Failed to send message");
    println!("Response: {:?}", response);
    // Assert the response
    assert_eq!(response.role(), "assistant");
    assert_eq!(response.model(), "claude-3-haiku-20240307");
    assert_eq!(response.stop_reason(), "end_turn");
    assert_eq!(response.usage().input_tokens, 18);
    assert!(response.usage().output_tokens > 0);
    assert!(!response.first_message().is_empty());
}

#[tokio::test]
async fn test_send_message_openai() {
    dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set.");
    let client_type = ClientLlm::OpenAI;
    let mut client = LlmClient::new(client_type, api_key);

    let response = client
        .request()
        .model("gpt-4o")
        .user_message("Hello, GPT!")
        .max_tokens(100)
        .temperature(1.0)
        .system_prompt("You are a haiku assistant.") // optional
        .send()
        .await
        .expect("Failed to send message");
    println!("Response: {:?}", response);
    // Assert the response
    assert_eq!(response.role(), "assistant");
    assert!(response.model().starts_with("gpt-4o"));
    assert_eq!(response.stop_reason(), "stop");
    assert_eq!(response.usage().input_tokens, 22);
    assert!(response.usage().output_tokens > 0);
    assert!(!response.first_message().is_empty());
}


#[tokio::test]
async fn test_invalid_api_key() {
    let api_key = "i am invalid".to_string();
    let client_type = ClientLlm::Anthropic;
    let mut client = LlmClient::new(client_type, api_key);

    let response = client
        .request()
        .user_message("Hello, Claude!")
        .send()
        .await;

    assert!(matches!(response, Err(ApiError::ClientError(_))));
}