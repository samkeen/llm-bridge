use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestBody {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContentBlock {
    pub text: String,
    #[serde(rename = "type")]
    pub block_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseMessage {
    pub id: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
}