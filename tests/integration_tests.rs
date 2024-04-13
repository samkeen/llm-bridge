use dotenv::dotenv;
use llm_api_adapter::client::AnthropicClient;
use llm_api_adapter::models::{Message};


#[tokio::test]
async fn test_send_message() {
    dotenv().ok();
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set.");
    let client = AnthropicClient::new(api_key.to_string());

    let messages = vec![Message {
        role: "user".to_string(),
        content: "Hello, Claude!".to_string(),
    }];

    let response = client
        .send_message("claude-3-haiku-20240307", messages, 100, 1.0)
        .await
        .expect("Failed to send message");

    // Assert the response
    assert_eq!(response.role, "assistant");
    assert!(!response.content.is_empty());
    assert_eq!(response.content[0].block_type, "text");
    assert!(!response.content[0].text.is_empty());
}

#[tokio::test]
async fn test_chat() {
    dotenv().ok();
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set.");
    let client = AnthropicClient::new(api_key.to_string());

    let conversation = client
        .chat("claude-3-haiku-20240307", 100, 1.0)
        .send("Hello, Claude!")
        .await
        .expect("Failed to send message");

    println!("Last response: {}", conversation.last_response());
    println!("Dialog:\n{:?}", conversation.dialog());

    let conversation = conversation
        .add("How are you doing?")
        .await
        .expect("Failed to send message");

    println!("Last response: {}", conversation.last_response());
    println!("Dialog:\n{:?}", conversation.dialog());
}