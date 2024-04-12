use dotenv::dotenv;
use llm_api_adapter::client::AnthropicClient;
use llm_api_adapter::models::Message;


#[tokio::test]
async fn test_send_message_integration() {
    dotenv().ok();
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set.");
    // Create an instance of AnthropicClient with your API key
    let client = AnthropicClient::new(api_key.to_string());

    // Prepare the input messages
    let messages = vec![Message {
        role: "user".to_string(),
        content: "Hello, Claude!".to_string(),
    }];

    // Send a message using the client
    let response = client
        .send_message("claude-3-haiku-20240307", messages, 100, 1.0)
        .await
        .expect("Failed to send message");

    // Assert the response
    assert_eq!(response.role, "assistant");
    assert!(!response.content.is_empty());
}