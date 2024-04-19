# LLM API Adapter SDK for Rust

This is a Rust SDK for interacting with various Large Language Model (LLM) APIs, starting with the Anthropic API. It
allows you to
send messages and engage in conversations with language models.

## Features

- Send single messages to the Anthropic API and retrieve responses
- Engage in multi-turn conversations with Anthropic's language models
- Customize the model, max tokens, and temperature for each request
- Handle API errors and parse responses using Rust structs

## Supported APIs

Currently, this SDK only supports the Anthropic API. However, there are plans to add support for additional Language
Model APIs in the future. Stay tuned for updates!

## Installation

Add the following to your `Cargo.toml` file:

```toml
[dependencies]
llm-api-adapter = "0.1.0"
```

## Usage

First, make sure you have an API key from Anthropic. Set the API key as an environment variable
named `ANTHROPIC_API_KEY`.

### Sending a Single Message

To send a single message to the Anthropic API and retrieve the response:

```rust
use llm_api_adapter::client::AnthropicClient;
use llm_api_adapter::models::Message;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set.");
    let client = AnthropicClient::new(api_key.to_string());

    let messages = vec![Message {
        role: "user".to_string(),
        content: "Hello, Claude!".to_string(),
    }];

    let response = client
        .send_message("claude-3-haiku-20240307", messages, 100, 1.0)
        .await
        .expect("Failed to send message");

    print!("Response: {}", response.first_message());
    println!("Response:\n{:?}", response);
}
```

### Engaging in a Conversation

To start a conversation with Anthropic's language models:

```rust
use llm_api_adapter::client::AnthropicClient;

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set.");
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
```

## Documentation

For detailed documentation and more examples, please refer to the [API documentation](link_to_docs).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a
pull request.

