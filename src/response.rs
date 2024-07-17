use std::fmt;
use serde::{Deserialize, Serialize};



#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub(crate) struct OpenAIUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct AnthropicResponse {
    pub id: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Represents a block of content in the API response.
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum AnthropicContentBlock {
    /// Represents a text content block in the Anthropic API response.
    Text {
        /// The actual text content of the response.
        text: String,
        /// The type of the content block, always "text" for this variant.
        #[serde(rename = "type")]
        block_type: String,
    },
    /// Represents a tool use content block in the Anthropic API response.
    /// This is used when the model decides to use a tool.
    ToolUse {
        /// The type of the content block, always "tool_use" for this variant.
        #[serde(rename = "type")]
        block_type: String,
        /// A unique identifier for this tool use instance.
        id: String,
        /// The name of the tool being used.
        name: String,
        /// The input provided to the tool, represented as a JSON value.
        /// This allows for flexibility in the structure of tool inputs.
        input: serde_json::Value,
    },
}

/// Represents the response message received from an LLM API.
///
/// The `ResponseMessage` enum encapsulates the different response types from various LLM APIs,
/// providing a unified interface for accessing common fields and methods.
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ResponseMessage {
    Anthropic(AnthropicResponse),
    OpenAI(OpenAIResponse),
}

impl ResponseMessage {
    /// Returns the text content of the first message in the response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::response::{AnthropicResponse, ResponseMessage};
    /// let response = ResponseMessage::Anthropic(AnthropicResponse {
    ///     id: "".to_string(),
    ///     role: "".to_string(),
    ///     content: vec![],
    ///     model: "".to_string(),
    ///     stop_reason: "".to_string(),
    ///     stop_sequence: None,
    ///     usage: Default::default(),}
    /// );
    /// let first_message = response.first_message();
    /// println!("First message: {}", first_message);
    /// ```
    pub fn first_message(&self) -> String {
        match self {
            ResponseMessage::Anthropic(response) => {
                if let Some(content) = response.content.first() {
                    match content {
                        AnthropicContentBlock::Text { text, .. } => text.clone(),
                        AnthropicContentBlock::ToolUse { .. } => String::new(), // or handle tool use as needed
                    }
                } else {
                    String::new()
                }
            }
            ResponseMessage::OpenAI(response) => {
                if let Some(choice) = response.choices.first() {
                    choice.message.content.clone().unwrap_or_else(|| {
                        // If content is None, it might be a function call response
                        if let Some(tool_calls) = &choice.message.tool_calls {
                            if let Some(first_tool) = tool_calls.first() {
                                format!("Function call: {}", first_tool.function.name)
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        }
                    })
                } else {
                    String::new()
                }
            }
        }
    }

    pub fn tools(&self) -> Option<Vec<ToolResponse>> {
        match self {
            ResponseMessage::Anthropic(response) => {
                let tool_uses: Vec<ToolResponse> = response.content.iter()
                    .filter_map(|block| {
                        if let AnthropicContentBlock::ToolUse { id, name, input, .. } = block {
                            Some(ToolResponse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                if tool_uses.is_empty() { None } else { Some(tool_uses) }
            },
            ResponseMessage::OpenAI(response) => {
                let tool_calls: Vec<ToolResponse> = response.choices.iter()
                    .filter_map(|choice| choice.message.tool_calls.as_ref())
                    .flatten()
                    .map(|tool_call| ToolResponse {
                        id: tool_call.id.clone(),
                        name: tool_call.function.name.clone(),
                        input: serde_json::from_str(&tool_call.function.arguments).unwrap_or(serde_json::Value::Null),
                    })
                    .collect();
                if tool_calls.is_empty() { None } else { Some(tool_calls) }
            },
        }
    }

    /// Returns the role of the sender in the response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::response::{AnthropicResponse, ResponseMessage};
    /// let response = ResponseMessage::Anthropic(AnthropicResponse {
    ///     id: "".to_string(),
    ///     role: "".to_string(),
    ///     content: vec![],
    ///     model: "".to_string(),
    ///     stop_reason: "".to_string(),
    ///     stop_sequence: None,
    ///     usage: Default::default(),}
    /// );
    /// let role = response.role();
    /// println!("Role: {}", role);
    /// ```
    pub fn role(&self) -> &str {
        match self {
            ResponseMessage::Anthropic(response) => &response.role,
            ResponseMessage::OpenAI(response) => {
                if let Some(choice) = response.choices.first() {
                    &choice.message.role
                } else {
                    ""
                }
            }
        }
    }

    /// Returns the name of the model used for generating the response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::response::{AnthropicResponse, ResponseMessage};
    /// let response = ResponseMessage::Anthropic(AnthropicResponse {
    ///     id: "".to_string(),
    ///     role: "".to_string(),
    ///     content: vec![],
    ///     model: "".to_string(),
    ///     stop_reason: "".to_string(),
    ///     stop_sequence: None,
    ///     usage: Default::default(),}
    /// );
    /// let model = response.model();
    /// println!("Model: {}", model);
    /// ```
    pub fn model(&self) -> &str {
        match self {
            ResponseMessage::Anthropic(response) => &response.model,
            ResponseMessage::OpenAI(response) => &response.model,
        }
    }

    /// Returns the stop reason for the generated response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::response::{AnthropicResponse, ResponseMessage};
    /// let response = ResponseMessage::Anthropic(AnthropicResponse {
    ///     id: "".to_string(),
    ///     role: "".to_string(),
    ///     content: vec![],
    ///     model: "".to_string(),
    ///     stop_reason: "".to_string(),
    ///     stop_sequence: None,
    ///     usage: Default::default(),}
    /// );
    /// let stop_reason = response.stop_reason();
    /// println!("Stop reason: {}", stop_reason);
    /// ```
    pub fn stop_reason(&self) -> &str {
        match self {
            ResponseMessage::Anthropic(response) => &response.stop_reason,
            ResponseMessage::OpenAI(response) => {
                if let Some(choice) = response.choices.first() {
                    &choice.finish_reason
                } else {
                    ""
                }
            }
        }
    }

    /// Returns the usage information for the generated response.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llm_bridge::response::{AnthropicResponse, ResponseMessage};
    /// let response = ResponseMessage::Anthropic(AnthropicResponse {
    ///     id: "".to_string(),
    ///     role: "".to_string(),
    ///     content: vec![],
    ///     model: "".to_string(),
    ///     stop_reason: "".to_string(),
    ///     stop_sequence: None,
    ///     usage: Default::default(),}
    /// );
    /// let usage = response.usage();
    /// println!("Input tokens: {}", usage.input_tokens);
    /// println!("Output tokens: {}", usage.output_tokens);
    /// ```
    pub fn usage(&self) -> CommonUsage {
        match self {
            ResponseMessage::Anthropic(response) => CommonUsage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
            },
            ResponseMessage::OpenAI(response) => CommonUsage {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
            },
        }
    }
}

impl fmt::Display for ResponseMessage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ResponseMessage::Anthropic(response) => {
                write!(
                    f,
                    "ResponseMessage {{ id: {}, role: {}, content: {:?} }}",
                    response.id, response.role, response.content
                )
            }
            ResponseMessage::OpenAI(response) => {
                write!(
                    f,
                    "ResponseMessage {{ id: {}, object: {}, model: {}, choices: {:?} }}",
                    response.id, response.object, response.model, response.choices
                )
            }
        }
    }
}


/// Tokens represent the underlying cost to llm systems.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct CommonUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIChoice {
    pub index: usize,
    pub message: OpenAIMessage,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ToolResponse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunction,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenAIFunction {
    pub name: String,
    pub arguments: String,
}

#[cfg(test)]

mod tests {
    use super::*;
    use serde_json::json;
    use crate::response::{AnthropicContentBlock, AnthropicResponse};

    #[test]
    fn test_anthropic_response_deserialization() {
        let json_response = json!({
            "id": "msg_01KGgxCr7Lm9gi1kfaZWWJUs",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01RQ6pzGpMxBBCirxUcSBokz",
                    "name": "get_weather",
                    "input": {
                        "location": "San Francisco, CA",
                        "unit": "celsius"
                    }
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 406,
                "output_tokens": 73
            }
        });

        let response: AnthropicResponse = serde_json::from_value(json_response).unwrap();

        assert_eq!(response.id, "msg_01KGgxCr7Lm9gi1kfaZWWJUs");
        assert_eq!(response.role, "assistant");
        assert_eq!(response.model, "claude-3-haiku-20240307");
        assert_eq!(response.stop_reason, "tool_use");
        assert_eq!(response.stop_sequence, None);
        assert_eq!(response.usage.input_tokens, 406);
        assert_eq!(response.usage.output_tokens, 73);

        assert_eq!(response.content.len(), 1);
        if let AnthropicContentBlock::ToolUse { id, name, input, .. } = &response.content[0] {
            assert_eq!(id, "toolu_01RQ6pzGpMxBBCirxUcSBokz");
            assert_eq!(name, "get_weather");
            assert_eq!(input["location"], "San Francisco, CA");
            assert_eq!(input["unit"], "celsius");
        } else {
            panic!("Expected ToolUse content block");
        }
    }

    #[test]
    fn test_anthropic_response_text_content() {
        let json_response = json!({
            "id": "msg_text_example",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [
                {
                    "type": "text",
                    "text": "This is a text response."
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        });

        let response: AnthropicResponse = serde_json::from_value(json_response).unwrap();

        assert_eq!(response.id, "msg_text_example");
        assert_eq!(response.role, "assistant");
        assert_eq!(response.model, "claude-3-haiku-20240307");
        assert_eq!(response.stop_reason, "end_turn");

        assert_eq!(response.content.len(), 1);
        if let AnthropicContentBlock::Text { text, .. } = &response.content[0] {
            assert_eq!(text, "This is a text response.");
        } else {
            panic!("Expected Text content block");
        }
    }

    #[test]
    fn test_anthropic_response_mixed_content() {
        let json_response = json!({
            "id": "msg_mixed_example",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [
                {
                    "type": "text",
                    "text": "Here's the weather information:"
                },
                {
                    "type": "tool_use",
                    "id": "toolu_mixed_example",
                    "name": "get_weather",
                    "input": {
                        "location": "New York, NY",
                        "unit": "fahrenheit"
                    }
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 50,
                "output_tokens": 60
            }
        });

        let response: AnthropicResponse = serde_json::from_value(json_response).unwrap();

        assert_eq!(response.content.len(), 2);

        match &response.content[0] {
            AnthropicContentBlock::Text { text, .. } => {
                assert_eq!(text, "Here's the weather information:");
            },
            _ => panic!("Expected Text content block"),
        }

        match &response.content[1] {
            AnthropicContentBlock::ToolUse { id, name, input, .. } => {
                assert_eq!(id, "toolu_mixed_example");
                assert_eq!(name, "get_weather");
                assert_eq!(input["location"], "New York, NY");
                assert_eq!(input["unit"], "fahrenheit");
            },
            _ => panic!("Expected ToolUse content block"),
        }
    }

    #[test]
    fn test_openai_response_deserialization() {
        let json_response = json!({
            "id": "chatcmpl-9p5LSmflVqlG0Gk6ryp14XHKbNah8",
            "object": "chat.completion",
            "created": 1721962302,
            "model": "gpt-4o-2024-05-13",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_5dENonKES2CcyWt6yGAXPDtz",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\":\"San Francisco, CA\"}"
                                }
                            }
                        ]
                    },
                    "logprobs": null,
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 106,
                "completion_tokens": 17,
                "total_tokens": 123
            },
            "system_fingerprint": "fp_400f27fa1f"
        });

        let response: OpenAIResponse = serde_json::from_value(json_response).unwrap();

        assert_eq!(response.id, "chatcmpl-9p5LSmflVqlG0Gk6ryp14XHKbNah8");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1721962302);
        assert_eq!(response.model, "gpt-4o-2024-05-13");

        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.finish_reason, "tool_calls");

        let message = &choice.message;
        assert_eq!(message.role, "assistant");
        assert_eq!(message.content, None);

        assert!(message.tool_calls.is_some());
        let tool_calls = message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call.id, "call_5dENonKES2CcyWt6yGAXPDtz");
        assert_eq!(tool_call.call_type, "function");
        assert_eq!(tool_call.function.name, "get_weather");
        assert_eq!(tool_call.function.arguments, "{\"location\":\"San Francisco, CA\"}");

        assert_eq!(response.usage.prompt_tokens, 106);
        assert_eq!(response.usage.completion_tokens, 17);
        assert_eq!(response.usage.total_tokens, 123);

    }

    #[test]
    fn test_openai_response_tool_calls() {
        let json_response = json!({
            "id": "chatcmpl-9p5LSmflVqlG0Gk6ryp14XHKbNah8",
            "object": "chat.completion",
            "created": 1721962302,
            "model": "gpt-4o-2024-05-13",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_5dENonKES2CcyWt6yGAXPDtz",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\":\"San Francisco, CA\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 106,
                "completion_tokens": 17,
                "total_tokens": 123
            }
        });

        let response: OpenAIResponse = serde_json::from_value(json_response).unwrap();
        let response_message = ResponseMessage::OpenAI(response);

        if let Some(tools) = response_message.tools() {
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name, "get_weather");
            assert_eq!(tools[0].id, "call_5dENonKES2CcyWt6yGAXPDtz");

            let input: serde_json::Value = serde_json::from_str(&tools[0].input.to_string()).unwrap();
            assert_eq!(input["location"], "San Francisco, CA");
        } else {
            panic!("Expected tool calls, but none were found");
        }

        assert_eq!(response_message.stop_reason(), "tool_calls");
    }

    #[test]
    fn test_openai_response_no_tool_calls() {
        let json_response = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1721962302,
            "model": "gpt-4o-2024-05-13",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a regular response without tool calls."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        });

        let response: OpenAIResponse = serde_json::from_value(json_response).unwrap();
        let response_message = ResponseMessage::OpenAI(response);
        assert_eq!(response_message.tools(), None);
        assert_eq!(response_message.stop_reason(), "stop");
        assert_eq!(response_message.first_message(), "This is a regular response without tool calls.");
    }
}