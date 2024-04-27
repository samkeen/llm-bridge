use dotenv::dotenv;
use babel_bridge::client::AnthropicClient;
use babel_bridge::error::ApiError;
use babel_bridge::models::{Message};
use pretty_assertions::{assert_eq};


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
        .request()
        .model("claude-3-haiku-20240307")
        .messages(messages)
        .max_tokens(100)
        .temperature(1.0)
        .system_prompt("You are a haiku assistant.") // optional
        .send()
        .await
        .expect("Failed to send message");
    println!("Response: {}", response.first_message());
    println!("Response model: {}", response.model);
    println!("Stop reason: {}", response.stop_reason);
    println!("Stop sequence: {}", response.stop_sequence.unwrap_or("".to_string()));
    println!("Input tokens: {}", response.usage.input_tokens);
    println!("Output tokens: {}", response.usage.output_tokens);
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
        .chat("claude-3-haiku-20240307", 100, 1.0, None)
        .send("Hello, Claude!")
        .await
        .expect("Failed to send message");

    println!("Last response: {}", conversation.last_response());
    println!("Dialog:\n{:?}", conversation.dialog());
    let first_usage_tallies = conversation.usage_tallies();
    println!("Token usage Tallies:\n{:?}", first_usage_tallies);
    assert_eq!(conversation.dialog().len(), 2);

    let conversation = conversation
        .add("How are you doing?")
        .await
        .expect("Failed to send message");

    println!("Last response: {}", conversation.last_response());
    println!("Dialog:\n{:?}", conversation.dialog());
    let last_usage_tallies = conversation.usage_tallies();
    println!("Token usage Tallies:\n{:?}", last_usage_tallies);
    assert_eq!(conversation.dialog().len(), 4);
    assert!(last_usage_tallies.input_tokens > first_usage_tallies.input_tokens);
    assert!(last_usage_tallies.output_tokens > first_usage_tallies.output_tokens);
}

#[tokio::test]
async fn test_invalid_api_key() {
    let api_key = "i am invalid";
    let client = AnthropicClient::new(api_key.into());
    let messages = vec![Message {
        role: "user".to_string(),
        content: "Hello, Claude!".to_string(),
    }];

    let response = client
        .request()
        .messages(messages)
        .send()
        .await;

    assert!(matches!(response, Err(ApiError::ClientError(_))));
}