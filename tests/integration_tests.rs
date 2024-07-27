#[cfg(test)]
mod tests {
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
        if let ResponseMessage::OpenAI(_) = response {
            panic!("Expected Anthropic response, but received OpenAI response");
        }
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
        if let ResponseMessage::Anthropic(_) = response {
            panic!("Expected Anthropic response, but received Anthropic response");
        }
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


    use std::sync::{Arc, Mutex};
    use std::thread;
    use llm_bridge::response::ResponseMessage;
    use llm_bridge::tool::Tool;

    #[tokio::test]
    async fn test_llm_client_sync() {
        // Create an instance of LlmClient with a mock API key
        let api_key = "mock_api_key".to_string();
        let client_type = ClientLlm::Anthropic;
        let llm_client = LlmClient::new(client_type, api_key);

        // Wrap the LlmClient in a Mutex and then in an Arc
        let shared_client = Arc::new(Mutex::new(llm_client));

        // Create a vector to store the thread handles
        let mut handles = vec![];

        // Spawn multiple threads that use the shared LlmClient
        for _ in 0..4 {
            let client = shared_client.clone();
            let handle = thread::spawn(move || {
                // Use the shared LlmClient within each thread
                let mut client_guard = client.lock().unwrap();
                let request_builder = client_guard
                    .request()
                    .model("claude-3-haiku-20240307")
                    .user_message("Hello, Claude!")
                    .max_tokens(100)
                    .temperature(1.0)
                    .system_prompt("You are a haiku assistant.");
                // ... Perform additional operations with the RequestBuilder
            });
            handles.push(handle);
        }

        // Wait for all threads to finish
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
        // The test passes if all threads completed successfully without any Sync-related issues
    }

    #[tokio::test]
    async fn test_tool_use_anthropic() {
        
        dotenv().ok();
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY must be set.");
        let client_type = ClientLlm::Anthropic;
        let mut client = LlmClient::new(client_type, api_key);

        let tool = Tool::builder()
            .name("get_weather")
            .description("Get the current weather in a given location")
            .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
            .add_enum_parameter("unit", "The unit of temperature, either 'celsius' or 'fahrenheit'", false, vec!["celsius".to_string(), "fahrenheit".to_string()])
            .build()
            .expect("Failed to build tool");

        let response = client
            .request()
            .add_tool(tool)
            .model("claude-3-haiku-20240307")
            .user_message("What is the current weather in San Francisco, California")
            .max_tokens(100)
            .temperature(1.0)
            .system_prompt("You are a weather assistant.")
            .send()
            .await
            .expect("Failed to send message");

        assert_eq!(response.stop_reason(), "tool_use");

        if let Some(tools) = response.tools() {
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name, "get_weather");
            assert_eq!(tools[0].input["location"], "San Francisco, CA");
        } else {
            panic!("Expected tool use, but no tools were returned");
        }
    }

    #[tokio::test]
    async fn test_tool_use_gpt() {

        dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY must be set.");
        let client_type = ClientLlm::OpenAI;
        let mut client = LlmClient::new(client_type, api_key);

        let tool = Tool::builder()
            .name("get_weather")
            .description("Get the current weather in a given location")
            .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
            .add_enum_parameter("unit", "The unit of temperature, either 'celsius' or 'fahrenheit'", false, vec!["celsius".to_string(), "fahrenheit".to_string()])
            .build()
            .expect("Failed to build tool");

        let response = client
            .request()
            .add_tool(tool)
            .model("gpt-4o")
            .user_message("What is the current weather in San Francisco, California")
            .max_tokens(100)
            .temperature(1.0)
            .system_prompt("You are a weather assistant.")
            .send()
            .await
            .expect("Failed to send message");

        assert_eq!(response.stop_reason(), "tool_calls");

        if let Some(tools) = response.tools() {
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name, "get_weather");
            assert_eq!(tools[0].input["location"], "San Francisco, CA");
        } else {
            panic!("Expected tool use, but no tools were returned");
        }
    }
}