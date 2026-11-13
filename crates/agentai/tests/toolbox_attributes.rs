//! Tests for the options of the `#[tool(...)]` attribute.
#![cfg(feature = "macros")]

use agentai::tool::{Tool, ToolBox, ToolError, ToolResult, ToolReturn, toolbox};

struct StrictToolBox;

#[toolbox]
impl StrictToolBox {
    /// Strict tool with parameters.
    #[tool(strict)]
    fn strict_tool(
        &self,
        /// City name
        city: String,
        /// Optional unit
        unit: Option<String>,
    ) -> ToolResult {
        Ok(format!("{city} {unit:?}"))
    }

    /// Explicitly non-strict tool.
    #[tool(strict = false)]
    fn not_strict(&self) -> ToolResult {
        Ok("ok".to_string())
    }

    /// Renamed strict tool.
    #[tool(name = "renamed", strict = true)]
    fn renamed_strict(&self) -> ToolResult {
        Ok("ok".to_string())
    }

    /// Tool without options.
    #[tool]
    fn plain(&self) -> ToolResult {
        Ok("ok".to_string())
    }
}

fn find<'a>(tools: &'a [Tool], name: &str) -> &'a Tool {
    tools
        .iter()
        .find(|t| t.name.as_str() == name)
        .unwrap_or_else(|| panic!("tool '{name}' not found"))
}

#[test]
fn strict_option_sets_the_strict_flag() -> Result<(), ToolError> {
    let tools = StrictToolBox.tools_definitions()?;

    assert_eq!(find(&tools, "strict_tool").strict, Some(true));
    assert!(find(&tools, "strict_tool").schema.is_some());
    assert_eq!(find(&tools, "not_strict").strict, Some(false));
    assert_eq!(find(&tools, "renamed").strict, Some(true));
    assert_eq!(find(&tools, "plain").strict, None);

    Ok(())
}

struct CallToolBox;

#[toolbox]
impl CallToolBox {
    /// Returns the call id and the text.
    #[tool]
    fn echo_id(
        &self,
        call: &agentai::ToolCall,
        /// Text to echo
        text: String,
    ) -> ToolResult {
        Ok(format!("{} {text}", call.id))
    }
}

#[tokio::test]
async fn a_tool_call_parameter_gets_the_call_and_stays_out_of_the_schema() -> Result<(), ToolError>
{
    let tools = CallToolBox.tools_definitions()?;
    let schema = find(&tools, "echo_id").schema.clone().unwrap();
    assert_eq!(schema["properties"].as_object().unwrap().len(), 1);
    assert!(schema["properties"]["text"].is_object());
    assert_eq!(schema["required"], serde_json::json!(["text"]));

    let call = agentai::ToolCall::new("call_7", "echo_id", serde_json::json!({"text": "hi"}));
    let ToolReturn::Done(output) = CallToolBox.call_tool(&call).await? else {
        panic!("expected ToolReturn::Done");
    };
    assert_eq!(output.text().as_deref(), Some("call_7 hi"));

    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema, Debug)]
struct Point {
    x: i32,
}

#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema, Debug)]
enum Unit {
    Celsius,
}

struct JsonStringToolBox;

#[toolbox]
impl JsonStringToolBox {
    /// Takes complex arguments.
    #[tool]
    fn complex(
        &self,
        tags: Vec<String>,
        point: Option<Point>,
        unit: Unit,
        note: String,
    ) -> ToolResult {
        Ok(format!("{tags:?} {point:?} {unit:?} {note}"))
    }

    /// The same tool, with the parsing turned off.
    #[tool(parse_json_strings = false)]
    fn exact(&self, tags: Vec<String>) -> ToolResult {
        Ok(format!("{tags:?}"))
    }
}

struct ExactToolBox;

#[toolbox(parse_json_strings = false)]
impl ExactToolBox {
    /// Takes a list.
    #[tool]
    fn list(&self, tags: Vec<String>) -> ToolResult {
        Ok(format!("{tags:?}"))
    }

    /// Takes a list, with the parsing turned back on.
    #[tool(parse_json_strings = true)]
    fn lenient(&self, tags: Vec<String>) -> ToolResult {
        Ok(format!("{tags:?}"))
    }
}

async fn call(toolbox: &impl ToolBox, name: &str, arguments: serde_json::Value) -> ToolResult {
    let call = agentai::ToolCall::new("call_1", name, arguments);
    let ToolReturn::Done(output) = toolbox.call_tool(&call).await? else {
        panic!("expected ToolReturn::Done");
    };
    Ok(output.text().unwrap_or_default())
}

#[tokio::test]
async fn arrays_and_objects_sent_as_json_strings_are_parsed() {
    let output = call(
        &JsonStringToolBox,
        "complex",
        serde_json::json!({
            "tags": "[\"a\", \"b\"]",
            "point": "{\"x\": 1}",
            "unit": "Celsius",
            "note": "[1]",
        }),
    )
    .await
    .unwrap();
    // A string argument keeps its JSON-looking text
    assert_eq!(output, r#"["a", "b"] Some(Point { x: 1 }) Celsius [1]"#);
}

#[tokio::test]
async fn strings_that_are_not_json_of_the_type_stay_errors() {
    for tags in ["a, b", "{\"a\": 1}"] {
        let result = call(
            &JsonStringToolBox,
            "complex",
            serde_json::json!({"tags": tags, "unit": "Celsius", "note": ""}),
        )
        .await;
        assert!(
            matches!(result, Err(ToolError::InvalidArguments(_))),
            "{tags}"
        );
    }
}

#[tokio::test]
async fn parse_json_strings_can_be_turned_off_and_on() {
    let tags = serde_json::json!({"tags": "[\"a\"]"});
    for (toolbox, name, parsed) in [
        (&JsonStringToolBox as &dyn ToolBox, "exact", false),
        (&ExactToolBox, "list", false),
        (&ExactToolBox, "lenient", true),
    ] {
        let call = agentai::ToolCall::new("call_1", name, tags.clone());
        let result = toolbox.call_tool(&call).await;
        assert_eq!(result.is_ok(), parsed, "{name}");
    }
}
