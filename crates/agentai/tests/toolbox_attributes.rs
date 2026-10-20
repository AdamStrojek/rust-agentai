//! Tests for the options of the `#[tool(...)]` attribute.
#![cfg(feature = "macros")]

use agentai::tool::{Tool, ToolBox, ToolError, ToolResult, toolbox};

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
