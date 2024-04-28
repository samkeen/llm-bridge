# LLM Bridge

## LLM API Adapter SDK for Rust

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
llm-bridge = "x.x.x"
```

## Usage

First, make sure you have an API key from Anthropic. Set the API key as an environment variable
named `ANTHROPIC_API_KEY`.

### Sending a Single Message

To send a single message to the Anthropic API and retrieve the response:

```rust
use llm_bridge::client::{ClientLlm, LlmClient};
use llm_bridge::error::ApiError;

#[tokio::main]
async fn main() {
    let api_key = "WWWWWWWWWW".to_string();
    let client_type = ClientLlm::OpenAI;
    let mut client = LlmClient::new(client_type, api_key);

    let response = client
        .request()
        .user_message("Hello, GPT!")
        .send()
        .await
        .expect("Failed to send message");
    println!("Response: {:?}", response);
    // Assert the response
    assert_eq!(response.role(), "assistant");
    assert_eq!(response.usage().input_tokens, 18);
    assert!(response.usage().output_tokens > 0);
    assert!(!response.first_message().is_empty());
}
```

Another example Using Anthropic's API and overriding some of the defaults on the request

```rust
use llm_bridge::client::{ClientLlm, LlmClient};
use llm_bridge::error::ApiError;

#[tokio::main]
async fn main() {
    let api_key = "WWWWWWWWWW".to_string();
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
    assert_eq!(response.usage().input_tokens, 18);
    assert!(response.usage().output_tokens > 0);
    assert!(!response.first_message().is_empty());
}
```

## Documentation

For detailed documentation and more examples, please refer to
the [API documentation](https://docs.rs/llm-bridge/latest/llm_bridge/).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a
pull request.

