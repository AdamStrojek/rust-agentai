//! Model settings and run statistics, independent of the LLM client used inside `agentai`.

use genai::adapter::AdapterKind;
use serde::{Deserialize, Serialize};

/// Model options sent with every request: temperature, max tokens, tool choice, reasoning
/// effort, ...
///
/// Every field left as `None` (or empty) uses the provider's default.
///
/// ```
/// use agentai::{ChatOptions, ReasoningEffort, ToolChoice};
///
/// let options = ChatOptions::default()
///     .with_temperature(0.0)
///     .with_max_tokens(2000)
///     .with_tool_choice(ToolChoice::Auto)
///     .with_reasoning_effort(ReasoningEffort::Low);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatOptions {
    /// Sampling temperature. Lower is more deterministic.
    pub temperature: Option<f64>,
    /// Maximum number of tokens in one answer. Reasoning models count their reasoning too.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling: only tokens within this probability mass are considered.
    pub top_p: Option<f64>,
    /// The model stops writing when it produces one of these strings.
    pub stop_sequences: Vec<String>,
    /// Whether and which tools the model must call.
    pub tool_choice: Option<ToolChoice>,
    /// How much the model should reason, for models that support it.
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ChatOptions {
    /// Sets the temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum number of tokens in one answer.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets nucleus sampling (`top_p`).
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the stop sequences.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = stop_sequences;
        self
    }

    /// Sets the tool choice.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets the reasoning effort.
    pub fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    /// Converts to genai options.
    pub(crate) fn to_genai(&self) -> genai::chat::ChatOptions {
        genai::chat::ChatOptions {
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop_sequences: self.stop_sequences.clone(),
            tool_choice: self.tool_choice.clone().map(ToolChoice::into_genai),
            reasoning_effort: self.reasoning_effort.map(ReasoningEffort::into_genai),
            ..Default::default()
        }
    }
}

/// Whether and which tools the model must call. See [`ChatOptions::tool_choice`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides.
    Auto,
    /// The model must not call tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call the tool with this name.
    Tool(String),
}

impl ToolChoice {
    pub(crate) fn into_genai(self) -> genai::chat::ToolChoice {
        let choice = self;
        match choice {
            ToolChoice::Auto => genai::chat::ToolChoice::Auto,
            ToolChoice::None => genai::chat::ToolChoice::None,
            ToolChoice::Required => genai::chat::ToolChoice::Required,
            ToolChoice::Tool(name) => genai::chat::ToolChoice::Tool { name },
        }
    }
}

/// How much a model should reason. See [`ChatOptions::reasoning_effort`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    /// Little reasoning, fast answers.
    Low,
    /// Medium reasoning.
    Medium,
    /// A lot of reasoning.
    High,
    /// A reasoning budget in tokens, for providers that support one.
    Budget(u32),
}

impl ReasoningEffort {
    pub(crate) fn into_genai(self) -> genai::chat::ReasoningEffort {
        let effort = self;
        match effort {
            ReasoningEffort::Low => genai::chat::ReasoningEffort::Low,
            ReasoningEffort::Medium => genai::chat::ReasoningEffort::Medium,
            ReasoningEffort::High => genai::chat::ReasoningEffort::High,
            ReasoningEffort::Budget(tokens) => genai::chat::ReasoningEffort::Budget(tokens),
        }
    }
}

/// Tokens used by a run. See [`AgentResult::usage`](crate::AgentResult::usage).
///
/// Counts the provider does not report are `0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    /// Tokens sent to the model, including cached ones.
    pub input_tokens: u64,
    /// Input tokens read from the provider's prompt cache.
    pub cached_input_tokens: u64,
    /// Input tokens written to the provider's prompt cache.
    pub cache_write_tokens: u64,
    /// Tokens written by the model, including reasoning.
    pub output_tokens: u64,
    /// Output tokens spent on reasoning.
    pub reasoning_tokens: u64,
    /// Total tokens, as reported by the provider.
    pub total_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.total_tokens += other.total_tokens;
    }
}

impl Usage {
    pub(crate) fn from_genai(usage: &genai::chat::Usage) -> Self {
        fn count(value: Option<i32>) -> u64 {
            value.map_or(0, |v| v.max(0) as u64)
        }
        let prompt = usage.prompt_tokens_details.as_ref();
        let completion = usage.completion_tokens_details.as_ref();
        Self {
            input_tokens: count(usage.prompt_tokens),
            cached_input_tokens: count(prompt.and_then(|d| d.cached_tokens)),
            cache_write_tokens: count(prompt.and_then(|d| d.cache_creation_tokens)),
            output_tokens: count(usage.completion_tokens),
            reasoning_tokens: count(completion.and_then(|d| d.reasoning_tokens)),
            total_tokens: count(usage.total_tokens),
        }
    }
}

/// Why the model stopped writing. See [`AgentResult::stop_reason`](crate::AgentResult::stop_reason).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    /// The model finished its answer.
    Completed,
    /// The answer was cut off by the token limit.
    MaxTokens,
    /// The model stopped to call tools.
    ToolCall,
    /// The provider filtered the answer.
    ContentFilter,
    /// The model produced one of the stop sequences.
    StopSequence,
    /// Any other reason, as reported by the provider.
    Other(String),
}

impl StopReason {
    /// Returns `true` when the answer was cut off by the token limit.
    pub fn is_max_tokens(&self) -> bool {
        matches!(self, Self::MaxTokens)
    }
}

impl StopReason {
    pub(crate) fn from_genai(reason: genai::chat::StopReason) -> Self {
        use genai::chat::StopReason as G;
        match reason {
            G::Completed(_) => Self::Completed,
            G::MaxTokens(_) => Self::MaxTokens,
            G::ToolCall(_) => Self::ToolCall,
            G::ContentFilter(_) => Self::ContentFilter,
            G::StopSequence(_) => Self::StopSequence,
            G::Other(raw) => Self::Other(raw),
        }
    }
}

/// LLM provider for [`Agent::new_with_provider`](crate::Agent::new_with_provider).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Provider {
    /// OpenAI Chat Completions API, and any OpenAI-compatible API with a custom base URL.
    OpenAI,
    /// OpenAI Responses API.
    OpenAIResponses,
    /// Anthropic.
    Anthropic,
    /// Google Gemini.
    Gemini,
    /// Google Vertex AI.
    Vertex,
    /// OpenRouter.
    OpenRouter,
    /// Groq.
    Groq,
    /// DeepSeek.
    DeepSeek,
    /// Mistral.
    Mistral,
    /// xAI.
    Xai,
    /// Cohere.
    Cohere,
    /// Ollama.
    Ollama,
    /// Fireworks.
    Fireworks,
    /// Together AI.
    Together,
    /// Nebius.
    Nebius,
    /// Moonshot.
    Moonshot,
    /// Z.ai.
    Zai,
    /// Alibaba Cloud (DashScope).
    Aliyun,
}

impl Provider {
    pub(crate) fn into_genai(self) -> AdapterKind {
        let provider = self;
        match provider {
            Provider::OpenAI => AdapterKind::OpenAI,
            Provider::OpenAIResponses => AdapterKind::OpenAIResp,
            Provider::Anthropic => AdapterKind::Anthropic,
            Provider::Gemini => AdapterKind::Gemini,
            Provider::Vertex => AdapterKind::Vertex,
            Provider::OpenRouter => AdapterKind::OpenRouter,
            Provider::Groq => AdapterKind::Groq,
            Provider::DeepSeek => AdapterKind::DeepSeek,
            Provider::Mistral => AdapterKind::Mistral,
            Provider::Xai => AdapterKind::Xai,
            Provider::Cohere => AdapterKind::Cohere,
            Provider::Ollama => AdapterKind::Ollama,
            Provider::Fireworks => AdapterKind::Fireworks,
            Provider::Together => AdapterKind::Together,
            Provider::Nebius => AdapterKind::Nebius,
            Provider::Moonshot => AdapterKind::Moonshot,
            Provider::Zai => AdapterKind::Zai,
            Provider::Aliyun => AdapterKind::Aliyun,
        }
    }
}
