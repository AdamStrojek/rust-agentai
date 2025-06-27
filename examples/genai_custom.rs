use agentai::Agent;
use genai::{adapter::AdapterKind, resolver::{AuthData, Endpoint, ServiceTargetResolver}, ClientBuilder, ModelIden, ServiceTarget};
use anyhow::Result;
use genai::chat::ChatOptions;
use log::{info, LevelFilter};
use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};

const SYSTEM: &str = "You are helpful assistant";

#[tokio::main]
async fn main() -> Result<()> {
    TermLogger::init(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Auto,
    )?;
    info!("Starting AgentAI");

    // Remember about providing what model you want to use
    // Some models may require providing authorization details
    let model = std::env::var("AGENTAI_MODEL").unwrap_or("gemini-2.0-flash".to_string());

    let question = "Why sky is blue?";

   	let target_resolver = ServiceTargetResolver::from_resolver_fn(
        |service_target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
            let endpoint = Endpoint::from_static("https://models.github.ai/inference/");
            let auth = AuthData::from_env("AGENTAI_API_KEY");
            let ServiceTarget { model, .. } = service_target;
            let model = ModelIden::new(AdapterKind::OpenAI, model.model_name);
            Ok(ServiceTarget { endpoint, auth, model })
        },
    );
    let chat_options = ChatOptions::default().with_temperature(0.0).with_max_tokens(20);
    let client = ClientBuilder::default()
        .with_chat_options(chat_options)
        .with_service_target_resolver(target_resolver).build();

    info!("Question: {}", question);

    let mut agent = Agent::new_with_client(client, SYSTEM);

    let answer: String = agent.run(&model, question, None).await?;

    info!("Answer: {}", answer);

    Ok(())
}
