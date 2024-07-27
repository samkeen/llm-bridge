use serde_json::{json, Map, Value};
use std::collections::HashMap;



#[derive(Debug, Clone)]
pub struct Tool {
    name: String,
    description: String,
    parameters: HashMap<String, ToolParameter>,
}

#[derive(Debug, Clone)]
pub struct ToolParameter {
    parameter_type: String,
    description: String,
    required: bool,
    enum_values: Option<Vec<String>>,
}

pub struct ToolBuilder {
    name: Option<String>,
    description: Option<String>,
    parameters: HashMap<String, ToolParameter>,
}

impl ToolBuilder {
    
    pub fn new() -> Self {
        ToolBuilder {
            name: None,
            description: None,
            parameters: HashMap::new(),
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    pub fn add_parameter(
        mut self,
        name: &str,
        parameter_type: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.parameters.insert(
            name.to_string(),
            ToolParameter {
                parameter_type: parameter_type.to_string(),
                description: description.to_string(),
                required,
                enum_values: None,
            },
        );
        self
    }

    pub fn add_enum_parameter(
        mut self,
        name: &str,
        description: &str,
        required: bool,
        enum_values: Vec<String>,
    ) -> Self {
        self.parameters.insert(
            name.to_string(),
            ToolParameter {
                parameter_type: "string".to_string(),
                description: description.to_string(),
                required,
                enum_values: Some(enum_values),
            },
        );
        self
    }

    pub fn build(self) -> Result<Tool, String> {
        let name = self.name.ok_or("Tool name is required")?;
        let description = self.description.ok_or("Tool description is required")?;

        Ok(Tool {
            name,
            description,
            parameters: self.parameters,
        })
    }
}

impl Tool {
    pub fn builder() -> ToolBuilder {
        ToolBuilder::new()
    }

    pub fn to_anthropic_format(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        self.process_tool_input(&mut properties, &mut required);

        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": ["location"]
            }
        })
    }

    pub fn to_openai_format(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        self.process_tool_input(&mut properties, &mut required);

        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    }

    fn process_tool_input(&self, properties: &mut Map<String, Value>, required: &mut Vec<Value>) {
        for (name, param) in &self.parameters {
            let mut property = serde_json::Map::new();
            property.insert(
                "type".to_string(),
                Value::String(param.parameter_type.clone()),
            );
            property.insert(
                "description".to_string(),
                Value::String(param.description.clone()),
            );

            if let Some(enum_values) = &param.enum_values {
                property.insert(
                    "enum".to_string(),
                    Value::Array(
                        enum_values
                            .iter()
                            .map(|v| Value::String(v.clone()))
                            .collect(),
                    ),
                );
            }

            properties.insert(name.clone(), Value::Object(property));

            if param.required {
                required.push(Value::String(name.clone()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_builder() {
        let tool = Tool::builder()
            .name("get_weather")
            .description("Get the current weather in a given location")
            .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
            .add_enum_parameter("unit", "The unit of temperature to use", false, vec!["celsius".to_string(), "fahrenheit".to_string()])
            .build()
            .expect("Failed to build tool");

        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.description, "Get the current weather in a given location");
        assert_eq!(tool.parameters.len(), 2);

        let location_param = tool.parameters.get("location").expect("Location parameter not found");
        assert_eq!(location_param.parameter_type, "string");
        assert_eq!(location_param.description, "The city and state, e.g. San Francisco, CA");
        assert!(location_param.required);
        assert!(location_param.enum_values.is_none());

        let unit_param = tool.parameters.get("unit").expect("Unit parameter not found");
        assert_eq!(unit_param.parameter_type, "string");
        assert_eq!(unit_param.description, "The unit of temperature to use");
        assert!(!unit_param.required);
        assert_eq!(unit_param.enum_values, Some(vec!["celsius".to_string(), "fahrenheit".to_string()]));
    }

    #[test]
    fn test_tool_builder_missing_name() {
        let result = Tool::builder()
            .description("Get the current weather in a given location")
            .build();

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Tool name is required");
    }

    #[test]
    fn test_tool_builder_missing_description() {
        let result = Tool::builder()
            .name("get_weather")
            .build();

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Tool description is required");
    }

    #[test]
    fn test_to_anthropic_format() {
        let tool = Tool::builder()
            .name("get_weather")
            .description("Get the current weather in a given location")
            .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
            .add_enum_parameter("unit", "The unit of temperature, either 'celsius' or 'fahrenheit'", false, vec!["celsius".to_string(), "fahrenheit".to_string()])
            .build()
            .expect("Failed to build tool");

        let anthropic_format = tool.to_anthropic_format();

        let expected = json!({
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                  "type": "string",
                  "enum": ["celsius", "fahrenheit"],
                  "description": "The unit of temperature, either 'celsius' or 'fahrenheit'"
                }
              },
              "required": ["location"]
            }
        });

        assert_eq!(anthropic_format, expected);
    }

    #[test]
    fn test_to_openai_format() {
        let tool = Tool::builder()
            .name("get_current_weather")
            .description("Get the current weather in a given location")
            .add_parameter("location", "string", "The city and state, e.g. San Francisco, CA", true)
            .add_enum_parameter("format", "The temperature unit to use. Infer this from the users location.", true, vec!["celsius".to_string(), "fahrenheit".to_string()])
            .build()
            .expect("Failed to build tool");

        let openai_format = tool.to_openai_format();

        let expected = json!({
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            }
        });
        assert_eq!(openai_format["type"], expected["type"]);
        assert_eq!(openai_format["function"]["name"], expected["function"]["name"]);
        assert_eq!(openai_format["function"]["description"], expected["function"]["description"]);

        let actual_params = &openai_format["function"]["parameters"];
        let expected_params = &expected["function"]["parameters"];

        assert_eq!(actual_params["type"], expected_params["type"]);
        assert_eq!(actual_params["properties"], expected_params["properties"]);
        // sort required prior to comparison
        let mut actual_required: Vec<String> = actual_params["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let mut expected_required: Vec<String> = expected_params["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        actual_required.sort();
        expected_required.sort();
        assert_eq!(actual_required, expected_required);
    }
}