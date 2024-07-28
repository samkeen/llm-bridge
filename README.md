# LLM Bridge

## Rust SDK adapter for LLM APIs

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
    let api_key = "YOUR API KEY".to_string();
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
    let api_key = "YOUR API KEY".to_string();
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

### Tool Use (Function Calling)

LLM Bridge now supports tool use (also known as function calling in OpenAI's terminology). This feature allows you to define tools or functions that the LLM can use to perform specific tasks.

#### Creating a Tool

To create a tool, use the `Tool` builder:

```rust
#[tokio::main]
async fn main() {
    use llm_bridge::tool::Tool;

    let weather_tool = Tool::builder()
        .name("get_weather")
        .description("Get the current weather in a given location")
        .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
        .add_enum_parameter("unit", "The unit of temperature", false, vec!["celsius".to_string(), "fahrenheit".to_string()])
        .build()
        .expect("Failed to build tool");
}
```

#### Using a Tool with OpenAI

Here's an example of how to use a tool with OpenAI's GPT model
Note: If using an Anthropic LLM, the function definition and the handling of the response remains the same.
```rust
use llm_bridge::client::{ClientLlm, LlmClient};
use llm_bridge::tool::Tool;

#[tokio::main]
async fn main() {
    let api_key = "your_openai_api_key".to_string();
    let client_type = ClientLlm::OpenAI;
    let mut client = LlmClient::new(client_type, api_key);

    let weather_tool = Tool::builder()
        .name("get_weather")
        .description("Get the current weather in a given location")
        .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
        .add_enum_parameter("unit", "The unit of temperature", false, vec!["celsius".to_string(), "fahrenheit".to_string()])
        .build()
        .expect("Failed to build tool");

    let response = client
        .request()
        .add_tool(weather_tool)
        .model("gpt-4o")
        .user_message("What's the weather like in New York?")
        .system_prompt("You are a helpful weather assistant.")
        .send()
        .await
        .expect("Failed to send message");

    if let Some(tools) = response.tools() {
        for tool in tools {
            println!("Tool used: {}", tool.name);
            println!("Tool input: {:?}", tool.input);
        }
    } else {
        println!("No tools were used in this response");
    }

    println!("Response: {}", response.first_message());
}
```






In this example, we define a `get_weather` tool and add it to the request. The LLM may choose to use this tool 
if it determines that it needs weather information to answer the user's question.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a
pull request.

