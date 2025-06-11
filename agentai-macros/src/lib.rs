use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input,
    Ident,
    ItemImpl,
    ImplItem,
    FnArg,
    Pat,
    Attribute,
    Lit,
    Meta,
    Error,
    Expr,
    MetaNameValue,
};
use proc_macro2::{Span, TokenStream as TokenStream2};
use std::collections::HashSet;

/// Attribute macro applied to an `impl` block to derive the `ToolBox` trait implementation.
///
/// Methods within the `impl` block annotated with `#[tool]` will be exposed as tools.
/// The `#[tool]` attribute can optionally take a `name` argument to specify the tool name.
/// Doc comments (`///`) on `#[tool]` methods are used as the tool description.
/// Doc comments (`#[doc = "..."]`) on function parameters are moved to the generated parameter struct fields.
///
/// Requires `serde`, `serde_json`, `schemars`, `async-trait`, and `anyhow` as dependencies in your project.
/// Make sure `schemars` is enabled with the `derive` feature.
///
/// Example:
/// ```no_run
/// use async_trait::async_trait;
/// use serde::{Serialize, Deserialize};
/// use serde_json::Value;
/// use rust_agentai_macros::toolbox;
/// use rust_agentai::tool::ToolError; // Assuming ToolError is accessible
/// use anyhow::Result;
///
/// struct MyToolBox {
///     my_field: i32,
/// }
///
/// #[toolbox]
/// impl MyToolBox {
///     // Constructor - not a tool as it's not #[tool]
///     pub fn new() -> Self {
///         Self { my_field: 69 }
///     }
///
///     /// This is the docstring for tool_one.
///     /// It demonstrates accessing a field.
///     #[tool]
///     async fn tool_one(&self) -> Result<String> {
///         Ok(format!("Result from tool one: {}", self.my_field))
///     }
///
///     /// This tool takes a parameter with documentation.
///     #[tool]
///     async fn tool_two(&self, #[doc = "The input string."] input: String) -> Result<String> {
///         Ok(format!("Tool two received: {}", input))
///     }
///
///     /// This tool has an altered name and takes a parameter without documentation.
///     #[tool(name = "my_special_tool")]
///     fn tool_three(&self, value: i32) -> Result<String> {
///         Ok(format!("Result from tool three with special name and value: {}", value))
///     }
///
///     /// This is a sync tool.
///     #[tool]
///     fn tool_sync(&self) -> Result<String> {
///          Ok("This is a synchronous tool result".to_string())
///     }
///
///     // This method will not be exposed as a tool
///     pub fn helper_method(&self) -> i32 {
///         42
///     }
/// }
///
/// // The macro generates the `impl ToolBox for MyToolBox` block
/// // and parameter structs like ToolTwoParams, ToolThreeParams, ToolSyncParams
/// ```
#[proc_macro_attribute]
pub fn toolbox(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_impl = parse_macro_input!(item as ItemImpl);

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

    let mut found_tools = HashSet::new();

    // Iterate over the methods to process the #[tool] attributes
    for item in &item_impl.items {
        if let ImplItem::Fn(method) = item {
            // Find the #[tool] attribute
            let tool_attr_option = method.attrs.iter().find(|attr| attr.path().is_ident("tool"));

            if let Some(tool_attr) = tool_attr_option {
                let fn_name = &method.sig.ident;
                let original_fn_name_str = fn_name.to_string();
                let mut tool_name = original_fn_name_str.clone();

                // Parse the #[tool] attribute for name = "..." using parse_args_with with Meta
                let parser = syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated;
                let args = match tool_attr.parse_args_with(parser) {
                    Ok(args) => args,
                    Err(e) => return Error::new_spanned(tool_attr.to_token_stream(), format!("Failed to parse tool attribute arguments: {}", e)).to_compile_error().into(),
                };

                // Iterate over the parsed Meta items to find 'name'. Allow #[tool] or #[tool(name = "...")].
                let mut name_arg_found = false;
                for arg_meta in args {
                    if let Meta::NameValue(name_value) = arg_meta {
                        if name_value.path.is_ident("name") {
                            if name_arg_found {
                                // Error: Duplicate 'name' argument
                                return Error::new_spanned(name_value.to_token_stream(), "Duplicate \'name\' argument in tool attribute").to_compile_error().into();
                            }
                            if let Expr::Lit(expr_lit) = &name_value.value {
                                if let Lit::Str(lit_str) = &expr_lit.lit {
                                    tool_name = lit_str.value();
                                    name_arg_found = true;
                                } else {
                                    // Error: Expected string literal for name
                                    return Error::new_spanned(expr_lit.to_token_stream(), "Expected string literal for tool name").to_compile_error().into();
                                }
                            } else {
                                // Error: Expected literal value for name
                                return Error::new_spanned(name_value.value.to_token_stream(), "Expected literal value for tool name").to_compile_error().into();
                            }
                        } else {
                             // Error: If arguments are present, they must be 'name = "..."'
                             return Error::new_spanned(name_value.path.to_token_stream(), "Expected only \'name\' argument in tool attribute").to_compile_error().into();
                        }
                    } else {
                         // Error: If arguments are present, they must be 'name = "..."'
                         return Error::new_spanned(arg_meta.to_token_stream(), "Expected name = \\\"...\\\" in tool attribute").to_compile_error().into();
                    }
                }

                // Check for duplicate tool names AFTER determining the final tool_name
                if !found_tools.insert(tool_name.clone()) {
                     return Error::new_spanned(tool_attr.to_token_stream(), format!("Duplicate tool name found: {}", tool_name)).to_compile_error().into();
                }


                // Extract doc comments for description from #[doc = "..."] attributes (handles /// and /* */)
                let description = method.attrs.iter()
                    .filter_map(|attr| {
                        // Use attr.meta field directly and match Result
                        if attr.path().is_ident("doc") {
                             match attr.meta.clone() { // Clone meta to consume in match
                                 Meta::NameValue(MetaNameValue { path, value: Expr::Lit(expr_lit), .. }) if path.is_ident("doc") => {
                                      match expr_lit.lit { // Access lit field
                                         Lit::Str(lit_str) => {
                                             // Remove leading slashes, stars, and whitespace
                                             Some(lit_str.value().trim_start_matches(|c: char| c == '/' || c == '*' || c.is_whitespace()).to_string())
                                         }
                                         _ => None, // Not a string literal
                                     }
                                 },
                                 _ => None, // Not a #[doc = ...] attribute or error
                             }
                        } else { None }
                    })
                    .collect::<Vec<String>>()
                    .join("\n");

                let description_token = if description.trim().is_empty() {
                    quote! { None }
                } else {
                    let desc = description.trim().to_string();
                    quote! { Some(#desc.to_string()) }
                };

                // Generate parameter struct
                use heck::ToUpperCamelCase;
                let params_struct_name = Ident::new(&format!("{}Params", original_fn_name_str.to_upper_camel_case()), fn_name.span());
                let mut param_fields = TokenStream2::new();
                let mut param_names: Vec<Ident> = vec![]; // Use Ident for parameter names


                // Skip `&self` or `&mut self` receiver
                for arg in method.sig.inputs.iter().filter(|arg| !matches!(arg, FnArg::Receiver(_))) {
                     if let FnArg::Typed(pat_type) = arg {
                        let pat = &pat_type.pat;
                        let ty = &pat_type.ty;

                        // Collect doc attributes from the parameter
                        let doc_attrs: Vec<Attribute> = pat_type.attrs.iter()
                            .filter(|attr| attr.path().is_ident("doc"))
                            // Clone the attributes we need to include in the generated struct
                            .cloned()
                            .collect();

                        if let Pat::Ident(pat_ident) = &**pat {
                            let arg_name = &pat_ident.ident;
                            param_fields.extend(quote! {
                                #(#doc_attrs)* // Include doc attributes here
                                pub #arg_name: #ty,
                            });
                            param_names.push(arg_name.clone());
                        } else {
                            // Handle other patterns if necessary, or return an error
                            return Error::new_spanned(pat.to_token_stream(), "Tool function parameters must be simple identifiers").to_compile_error().into();
                        }
                     } else {
                        // Should not happen after filtering Receiver, but good practice
                         return Error::new_spanned(arg.to_token_stream(), "Unexpected function argument type in tool method").to_compile_error().into();
                      }
                }

                let params_struct_definition = if param_fields.is_empty() {
                    // No parameters other than self, generate an empty struct
                     quote! {
                        // Parameters struct for #original_fn_name_str
                        #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]

                        // Allow dead code if the struct is only used by the generated macro code
                        #[allow(dead_code)]
                         #[allow(clippy::all)] // Allow various lints for generated code
                        struct #params_struct_name {}
                    }
                } else {
                    // Parameters exist, generate struct with fields
                    quote! {
                        // Parameters struct for #original_fn_name_str
                        #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
                        // Allow dead code if the struct is only used by the generated macro code
                        #[allow(dead_code)]
                         #[allow(clippy::all)] // Allow various lints for generated code
                         struct #params_struct_name {
                            #param_fields
                         }
                     }
                 };
                 // Always generate the struct definition, even if empty, so schemars::schema_for! works consistently.
                 generated_code.extend(params_struct_definition);


                // Add to tool definitions
                let schema_token = if param_fields.is_empty() {
                    quote! { None }
                } else {
                    // Use the generated parameter struct name for schemars::schema_for!
                    quote! { Some(serde_json::to_value(schemars::schema_for!(#params_struct_name)).expect("Failed to serialize schema")) }
                };

                tool_definitions.extend(quote! {
                    crate::tool::Tool {
                        name: #tool_name.to_string(),
                        description: #description_token,
                        schema: #schema_token,
                    },
                });

                // Add to match arms for call_tool
                let param_assignments = param_names.iter().map(|name| {
                    // Direct access to fields of the deserialized params struct
                    quote! { params.#name }
                }).collect::<TokenStream2>();

                // Determine the method call, including parameters
                let method_call = if param_names.is_empty() {
                    quote! { self.#fn_name() }
                } else {
                    quote! { self.#fn_name(#param_assignments) }
                };

                // Add .await if the original function was async
                let await_if_async = if method.sig.asyncness.is_some() {
                    quote! {.await}
                } else {
                    quote! {}
                };


                let call_body = if param_fields.is_empty() {
                    // No parameters to deserialize, just call the method
                    quote! {
                         #method_call #await_if_async .map_err(|e| {
                             // Map any error from the tool function to ToolError::ExecutionError
                             // Consider logging the original error `e` here for debugging
                             #[cfg(feature = "log-errors")] // Use a feature flag for logging
                             eprintln!("Tool execution error for '{}': {:?}", #tool_name, e);
                             crate::tool::ToolError::ExecutionError
                         })
                     }
                } else {
                    // Deserialize parameters and call the method
                    quote! {
                        let params: #params_struct_name = serde_json::from_value(parameters)
                            .map_err(|e| {
                                // Map deserialization error to ToolError::ExecutionError
                                // Consider logging the deserialization error `e` here
                                #[cfg(feature = "log-errors")] // Use a feature flag for logging
                                eprintln!("Tool parameter deserialization error for '{}': {:?}", #tool_name, e);
                                crate::tool::ToolError::ExecutionError
                            })?; // Use ? to propagate deserialization error

                        #method_call #await_if_async .map_err(|e| {
                            // Map any error from the tool function to ToolError::ExecutionError
                            // Consider logging the original error `e` here for debugging
                            #[cfg(feature = "log-errors")] // Use a feature flag for logging
                            eprintln!("Tool execution error for '{}': {:?}", #tool_name, e);
                            crate::tool::ToolError::ExecutionError
                        })
                    }
                };

                match_arms.extend(quote! {
                    #tool_name => {
                        #call_body
                    },
                });
            }
        }
    }


    // Generate the ToolBox implementation
    let toolbox_impl = quote! {
        // Need necessary imports for the generated code
        #[allow(unused_imports)] // Allow unused imports if no tools or params are generated
        use serde::{Serialize, Deserialize};
        #[allow(unused_imports)]
        use schemars::{self, JsonSchema};

        #[async_trait::async_trait]
        impl crate::tool::ToolBox for #struct_ident {

            fn tools_definitions(&self) -> Result<Vec<crate::tool::Tool>, crate::tool::ToolError> {
                Ok(vec![
                    #tool_definitions
                ])
            }

            async fn call_tool(&self, tool_name: String, parameters: serde_json::Value) -> Result<String, crate::tool::ToolError> {
                 match tool_name.as_str() {
                     #match_arms
                     _ => {
                         Err(crate::tool::ToolError::NoToolFound(tool_name))
                     }
                 }
            }
        }
    };

    // Combine generated code and the original impl block
    let final_code = quote! {
        #generated_code

        #toolbox_impl

        #item_impl // Keep the original impl block
    };

    final_code.into()
}

#[proc_macro_attribute]
/// Attribute to mark a method within a ToolBox implementation as a callable tool.
///
/// This macro parses the attribute arguments (like `name = "..."`) and the annotated
/// function definition but primarily serves to register the `#[tool]` attribute
/// name with the compiler. The actual tool processing (schema generation,
/// call dispatch) is handled by the `#[toolbox]` macro applied to the
/// `impl` block.
///
/// Expected arguments:
/// - `name = "tool_name"`: Specifies the unique name for the tool. This is required.
pub fn tool(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // The #[toolbox] macro will parse and process the attribute arguments.
    // This macro only needs to make the attribute name known to the compiler.

    // Pass the annotated item (the function) through unmodified.
    // The #[toolbox] macro will read and process this function later.
    item
}
