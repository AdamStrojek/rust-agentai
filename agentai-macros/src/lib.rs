use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, Error, Expr, FnArg, Ident, ImplItem, ItemImpl, Lit, Meta, MetaNameValue, Pat
};
use std::collections::HashSet;
use heck::ToUpperCamelCase;

/// # Macro for Generating `ToolBox` Implementations
///
/// The `#[toolbox]` attribute macro streamlines the process of implementing the `ToolBox` trait
/// for a given struct. By applying this macro to an `impl` block, you can designate specific
/// methods as "tools" that are discoverable and callable.
///
/// This macro handles the following:
/// - **Tool Definition**: It automatically generates metadata for each tool, including its name,
///   description, and a JSON schema for its parameters.
/// - **Dispatch Logic**: It creates the necessary logic to dispatch calls to the appropriate tool method.
///
/// ## Prerequisites
///
/// Ensure your `Cargo.toml` includes the following dependencies:
///
/// ```toml
/// serde = { version = "1.0", features = ["derive"] }
/// serde_json = "1.0"
/// schemars = { version = "0.9", features = ["derive"] }
/// async-trait = "0.1"
/// ```
///
/// You must also import the necessary components from the `agentai::tool` module:
///
/// ```no_run
/// use agentai::tool::{Tool, ToolBox, ToolError, toolbox};
/// ```
///
/// ## Usage Guide
///
/// ### 1. Defining Your ToolBox Struct
///
/// First, define a struct that will serve as your `ToolBox`. This struct can hold state,
/// such as API keys or a shared HTTP client, which can be accessed by your tools.
///
/// The `impl` block for this struct must be annotated with `#[toolbox]`.
///
/// ```no_run
/// struct MyToolBox {
///     api_key: String,
/// }
///
/// #[toolbox]
/// impl MyToolBox {
///     pub fn new(api_key: String) -> Self {
///         Self { api_key }
///     }
///
///     // Tool methods will be defined here
/// }
/// ```
///
/// ### 2. Exposing Methods as Tools with `#[tool]`
///
/// To expose a method as a tool, annotate it with the `#[tool]` attribute. This attribute is a marker
/// and does not need to be imported. Both synchronous and asynchronous methods are supported.
///
/// #### 2.1. Default Behavior
///
/// - **Tool Name**: The tool's name is inferred from the method's name. It must be unique within the toolbox.
/// - **Tool Description**: The method's documentation comments (`///` or `#[doc = "..."]`) are used as the tool's description.
/// - **Parameter Schema**: A JSON schema is automatically generated from the method's parameters.
///
/// #### 2.2. Requirements and Limitations
///
/// - **Method Receiver**: Exposed tools must be methods that take `&self` as the first argument. Static methods are not supported.
/// - **Return Type**: The return type must be `Result<String, ToolError>`.
/// - **Serializable Parameters**: All method parameters must be (de)serializable by `serde`.
///
/// ### 3. Advanced Configuration
///
/// The `#[tool(...)]` attribute gives you broad control over the configuration of declared tools.
/// You can change any of the options using `name=value` pairs. The following options are supported:
/// - `name`: Overrides the default tool name. This name must be unique within the toolbox.
///
/// ### 4. Tool Arguments
/// The tool's schema is generated based on the method's arguments, which is why they must be serializable.
/// This is primarily syntactic sugar, as all arguments are copied into a new helper structure as serializable fields.
/// This struct derives `serde::Serialize`, `serde::Deserialize`, and `schemars::JsonSchema` to handle argument
/// serialization, deserialization, and schema generation.
///
/// All attributes for the arguments will be moved from the method implementation to the newly generated arguments structure.
/// This allows you to not only provide documentation for the purpose of an argument but also to modify its behavior using
/// `serde` or `schemars` attributes. For more information, refer to the following pages:
/// - [serde](https://serde.rs/field-attrs.html)
/// - [schemars](https://graham.cool/schemars/examples/3-schemars_attrs/)
///
/// # Examples
///
/// ```no_run
/// use agentai::tool::{Tool, ToolBox, ToolError, toolbox};
///
/// struct MyToolBox {
///     my_field: i32,
/// }
///
/// #[toolbox]
/// impl MyToolBox {
///     pub fn new() -> Self {
///         Self { my_field: 69 }
///     }
///
///     /// This tool demonstrates accessing a field on the struct.
///     #[tool]
///     async fn tool_one(&self) -> Result<String, ToolError> {
///         Ok(format!("Result from tool one: {}", self.my_field))
///     }
///
///     /// This tool takes a parameter with documentation.
///     #[tool]
///     async fn tool_two(&self, #[doc = "The input string."] input: String) -> Result<String, ToolError> {
///         Ok(format!("Tool two received: {}", input))
///     }
///
///     /// This tool has an altered name.
///     #[tool(name = "my_special_tool")]
///     fn tool_three(
///         &self,
///         /// You can use both methods of providing documentation for an argument
///         value: i32
///     ) -> Result<String, ToolError> {
///         Ok(format!("Result from tool three with special name and value: {}", value))
///     }
///
///     /// This is a sync tool method example.
///     #[tool]
///     fn tool_sync(&self) -> Result<String, ToolError> {
///          Ok("This is a synchronous tool result".to_string())
///     }
///
///     // This method will not be exposed as a tool because it lacks the #[tool] attribute.
///     pub fn helper_method(&self) -> i32 {
///         42
///     }
/// }
/// ```
///
/// ## Generated Code
///
/// The `#[toolbox]` macro generates the following:
///
/// 1.  **Parameter Structs**: For each tool with parameters, a private struct is generated
///     (e.g., `ToolTwoParams`). These structs derive `serde::Serialize`, `serde::Deserialize`,
///     and `schemars::JsonSchema` to manage parameter handling and schema generation.
///
/// 2.  **`ToolBox` Implementation**: It generates the `impl ToolBox for YourStruct` block.
///     -   **`tools_definitions`**: This method returns a `Vec<Tool>`, providing the metadata for each exposed tool.
///     -   **`call_tool`**: This method acts as a dispatcher. It matches the `tool_name`,
///         deserializes the JSON `parameters` into the corresponding parameter struct,
///         and invokes the actual method.
#[proc_macro_attribute]
pub fn toolbox(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the original impl block
    let mut item_impl = parse_macro_input!(item as ItemImpl);

    let struct_name = &item_impl.self_ty;
    let struct_ident = match &**struct_name {
        syn::Type::Path(type_path) => {
            type_path.path.get_ident().expect("Expected an identifier for the struct")
        }
        _ => return Error::new(Span::call_site(), "toolbox! macro only supports impl blocks for structs").to_compile_error().into(),
    };

    let mut generated_code = TokenStream2::new();
    let mut tool_definitions = TokenStream2::new();
    let mut match_arms = TokenStream2::new();

    // TODO: Maybe we should use BTreeHash to preserve order of tools?
    let mut found_tools = HashSet::new();

    // Pass 1: Collect information for tool definitions and call dispatch
    // We iterate over a reference here because we need the original items again in Pass 2
    for item in item_impl.items.iter_mut() {
        if let ImplItem::Fn(ref mut method) = item {
            // Find the #[tool] attribute
            if let Some(tool_attr) = method.attrs.clone().iter().find(|attr| attr.path().is_ident("tool")) {
                // Remove #[tool] attribute
                // #[tool] is used only to mark functions that will be converted into tools
                method.attrs.retain(|attr| !attr.path().is_ident("tool"));

                let fn_name_sig = &method.sig.ident;
                let fn_name = fn_name_sig.to_string();
                let mut tool_name = fn_name.clone();

                // Parse the #[tool] attribute for name = "..." using parse_args_with with Meta
                let mut name_arg_found = false;
                let parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
                if let Ok(args) = tool_attr.parse_args_with(parser) {
                    // Iterate over the parsed Meta items to find 'name'. #[tool(name = "...")]
                    for arg_meta in args {
                        match arg_meta {
                            Meta::NameValue(name_value) if name_value.path.is_ident("name") => {
                                if name_arg_found {
                                    // Error: Duplicate 'name' argument
                                    return Error::new_spanned(name_value.to_token_stream(), "Duplicate 'name' argument in tool attribute").to_compile_error().into();
                                }
                                let Expr::Lit(expr_lit) = &name_value.value else {
                                    // Error: Expected literal value for name
                                    return Error::new_spanned(name_value.value.to_token_stream(), "Expected literal value for tool name").to_compile_error().into();
                                };
                                let Lit::Str(lit_str) = &expr_lit.lit else {
                                    // Error: Expected string literal for name
                                    return Error::new_spanned(expr_lit.to_token_stream(), "Expected string literal for tool name").to_compile_error().into();
                                };
                                tool_name = lit_str.value();
                                name_arg_found = true;
                            },
                            _ => {
                                // Error: If arguments are present, they must be 'name = "..."'
                                return Error::new_spanned(arg_meta.to_token_stream(), "Expected name = \"...\" in tool attribute").to_compile_error().into();
                            }
                        };
                    }
                }

                // Check for duplicate tool names AFTER determining the final tool_name
                if !found_tools.insert(tool_name.clone()) {
                     return Error::new_spanned(tool_attr.to_token_stream(), format!("Duplicate tool name found: {}", tool_name)).to_compile_error().into();
                }

                // Extract doc comments for description from #[doc = "..."] attributes (handles /// and /* */) from method
                let description = method.attrs.iter()
                    .filter_map(|attr|
                        match attr.meta.clone() {
                            Meta::NameValue(MetaNameValue { path, value: Expr::Lit(expr_lit), .. }) if path.is_ident("doc") => {
                                match expr_lit.lit {
                                    Lit::Str(lit_str) => {
                                        // Remove leading slashes, stars, and whitespace
                                        Some(lit_str.value().trim().trim_start_matches(|c: char| c == '/' || c == '*' || c.is_whitespace()).to_string())
                                    }
                                    _ => None, // Not a string literal
                                }
                            },
                            _ => None, // Not a #[doc = ...] attribute or error
                        }
                    )
                    .collect::<Vec<String>>()
                    .join("\n");

                let description_token = if description.trim().is_empty() {
                    quote! { None }
                } else {
                    let desc = description.trim().to_string();
                    quote! { Some(#desc.to_string()) }
                };

                // Generate parameter struct
                let params_struct_name = Ident::new(&format!("{}Params", fn_name.to_upper_camel_case()), fn_name_sig.span());
                let mut param_fields = TokenStream2::new();
                let mut param_assignments = TokenStream2::new();

                for arg in method.sig.inputs.iter_mut() {
                    // self attribute are type FnArg::Receiver()
                    if let FnArg::Typed(ref mut pat_type) = arg {
                        // #[doc = "Documentation"]    // < pat_type.attrs
                        // attribute: Type,            // < pat_type.pat: pat_type.ty
                        // ...
                        let ty = pat_type.ty.clone();

                        // Clone all attributes that will be moved to new structure
                        let attrs = pat_type.attrs.clone();

                        // Clean attributes for tool definition
                        pat_type.attrs.clear();

                        let Pat::Ident(ref pat_ident) = *pat_type.pat else {
                            // Handle other patterns if necessary, or return an error
                            return Error::new_spanned(pat_type.pat.to_token_stream(), "Tool function parameters must be simple identifiers").to_compile_error().into();
                        };

                        let arg_name = &pat_ident.ident;
                        // TODO: Change pub to pub(crate), this structures will be used only inside generated code
                        param_fields.extend(quote! {
                            #(#attrs)* pub #arg_name: #ty,
                        });

                        param_assignments.extend(quote! {
                            params.#arg_name
                        });
                    }
                }

                if !param_fields.is_empty() {
                    generated_code.extend(quote! {
                        // Parameters struct for #original_fn_name_str
                        #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
                        #[allow(dead_code)]
                        #[allow(clippy::all)]
                        struct #params_struct_name {
                            #param_fields
                        }
                     });
                }

                // Add to tool definitions
                let schema_token = if param_fields.is_empty() {
                    quote! { None }
                } else {
                    // Use the generated parameter struct name for schemars::schema_for!
                    // quote! { Some(generate_tool_schema::<#params_struct_name>()) }
                    quote! {
                        Some({
                            let generator = ::schemars::generate::SchemaSettings::draft2020_12().with(|s| {
                                s.meta_schema = None;
                            }).into_generator();
                            generator.into_root_schema_for::<#params_struct_name>().into()
                        })
                    }
                };

                tool_definitions.extend(quote! {
                    Tool {
                        name: #tool_name.to_string(),
                        description: #description_token,
                        schema: #schema_token,
                    },
                });

                // Add to match arms for call_tool
                let mut method_call = TokenStream2::new();

                if !param_fields.is_empty(){
                    method_call.extend(quote! {
                        let params: #params_struct_name = serde_json::from_value(parameters)
                            .map_err(|e| {
                                eprintln!("Tool parameter deserialization error for '{}': {:?}", #tool_name, e);
                                ToolError::ExecutionError
                            })?;
                    });
                }

                method_call.extend(quote! { self.#fn_name_sig(#param_assignments) });
                if method.sig.asyncness.is_some() {
                    method_call.extend(quote! {.await});
                }

                method_call.extend(quote! { .map_err(|e| {
                    eprintln!("Tool execution error for '{}': {:?}", #tool_name, e);
                    ToolError::ExecutionError
                }) });

                match_arms.extend(quote! {
                    #tool_name => {
                        #method_call
                    },
                });
            }
        }
    }

    if found_tools.is_empty() {
        return Error::new(Span::call_site(), "No #[tool] definition in impl block").to_compile_error().into()
    }

    // Generate the ToolBox implementation
    let toolbox_impl = quote! {
        #[::async_trait::async_trait]
        impl ToolBox for #struct_ident {

            fn tools_definitions(&self) -> Result<Vec<Tool>, ToolError> {
                Ok(vec![
                    #tool_definitions
                ])
            }

            async fn call_tool(&self, tool_name: String, parameters: serde_json::Value) -> Result<String, ToolError> {
                 match tool_name.as_str() {
                     #match_arms
                     _ => {
                         Err(ToolError::NoToolFound(tool_name))
                     }
                 }
            }
        }
    };

    // Combine generated code, the ToolBox impl, and the modified original impl block
    let final_code = quote! {
        #item_impl

        #toolbox_impl

        #generated_code
    };

    final_code.into()
}
