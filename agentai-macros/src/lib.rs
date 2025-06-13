use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parse::Parser, parse_macro_input, punctuated::Punctuated, Attribute, Error, Expr, FnArg, Ident, ImplItem, ItemImpl, Lit, Meta, MetaNameValue, Pat
};
use std::collections::HashSet;
use heck::ToUpperCamelCase;

/// Attribute macro to generate a `ToolBox` implementation for a struct.
///
/// Apply this macro to an `impl` block for your struct. Any `async fn`
/// methods within this `impl` block marked with the `#[tool]` attribute
/// will be automatically exposed as tools.
///
/// The macro generates:
/// 1. A `serde::Serialize`, `serde::Deserialize`, and `schemars::JsonSchema`
///    struct for the parameters of each #[tool] function. Doc comments (`#[doc = "..."`)
///    on the parameters will be included as attributes on the struct fields.
/// 2. An implementation of the `ToolBox` trait for the struct.
///    - `tools_definitions` method that returns a list of `Tool` structs
///      based on the #[tool] methods and their documentation/schemas.
///    - `call_tool` method that dispatches calls to the appropriate #[tool]
///      method based on the tool name and deserializes the provided parameters.
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

    let mut found_tools = HashSet::new();

    // Pass 1: Collect information for tool definitions and call dispatch
    // We iterate over a reference here because we need the original items again in Pass 2
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

                // Iterate over the parsed Meta items to find 'name'. Allow #[tool] or #[tool(name = "...")] or #[tool()].
                let mut name_arg_found = false;
                for arg_meta in args {
                    if let Meta::NameValue(name_value) = arg_meta {
                        if name_value.path.is_ident("name") {
                            if name_arg_found {
                                // Error: Duplicate 'name' argument
                                return Error::new_spanned(name_value.to_token_stream(), "Duplicate 'name' argument in tool attribute").to_compile_error().into();
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
                             return Error::new_spanned(name_value.path.to_token_stream(), "Expected only 'name' argument in tool attribute").to_compile_error().into();
                         }
                    } else {
                         // Error: If arguments are present, they must be 'name = "..."'
                         return Error::new_spanned(arg_meta.to_token_stream(), "Expected name = \"...\" in tool attribute").to_compile_error().into();
                     }
                }

                // Check for duplicate tool names AFTER determining the final tool_name
                if !found_tools.insert(tool_name.clone()) {
                     return Error::new_spanned(tool_attr.to_token_stream(), format!("Duplicate tool name found: {}", tool_name)).to_compile_error().into();
                }


                // Extract doc comments for description from #[doc = "..."] attributes (handles /// and /* */) from method
                let description = method.attrs.iter()
                    .filter_map(|attr| {
                        if attr.path().is_ident("doc") {
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
                let params_struct_name = Ident::new(&format!("{}Params", original_fn_name_str.to_upper_camel_case()), fn_name.span());
                let mut param_fields = TokenStream2::new();
                let mut param_names: Vec<Ident> = vec![];

                // Skip `&self` or `&mut self` receiver
                for arg in method.sig.inputs.iter().filter(|arg| !matches!(arg, FnArg::Receiver(_))) {
                     if let FnArg::Typed(pat_type) = arg {
                        let pat = &pat_type.pat;
                        let ty = &pat_type.ty;

                        // Collect doc attributes from the parameter to include in the generated struct
                        let doc_attrs_for_struct: Vec<Attribute> = pat_type.attrs.iter()
                            .filter(|attr| attr.path().is_ident("doc"))
                            .cloned()
                            .collect();

                        if let Pat::Ident(pat_ident) = &**pat {
                            let arg_name = &pat_ident.ident;
                            param_fields.extend(quote! {
                               #(#doc_attrs_for_struct)* pub #arg_name: #ty,
                            });
                            param_names.push(arg_name.clone());
                        } else {
                            // Handle other patterns if necessary, or return an error
                            return Error::new_spanned(pat.to_token_stream(), "Tool function parameters must be simple identifiers").to_compile_error().into();
                        }
                     } else {
                          return Error::new_spanned(arg.to_token_stream(), "Unexpected function argument type in tool method").to_compile_error().into();
                       }
                }

                let params_struct_definition = if param_fields.is_empty() {
                     quote! {
                        // Parameters struct for #original_fn_name_str
                        #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
                        #[allow(dead_code)]
                         #[allow(clippy::all)]
                        struct #params_struct_name {}; // Add semicolon for empty struct
                    }
                } else {
                    quote! {
                        // Parameters struct for #original_fn_name_str
                        #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
                        #[allow(dead_code)]
                         #[allow(clippy::all)]
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
                            eprintln!("Tool execution error for '{}': {:?}", #tool_name, e);
                            ToolError::ExecutionError
                        })
                    }
                } else {
                    // Deserialize parameters and call the method
                    quote! {
                        let params: #params_struct_name = serde_json::from_value(parameters)
                            .map_err(|e| {
                                eprintln!("Tool parameter deserialization error for '{}': {:?}", #tool_name, e);
                                ToolError::ExecutionError
                            })?;
                        #method_call #await_if_async .map_err(|e| {
                            eprintln!("Tool execution error for '{}': {:?}", #tool_name, e);
                            ToolError::ExecutionError
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

    // Pass 2: Reconstruct item_impl removing #[doc] and #[tool] from tool methods
    // Iterate through the original items again to modify them for the final output
    let mut final_items = Vec::<ImplItem>::new();

    // We consume item_impl.items here
    for item in item_impl.items.into_iter() {
        if let ImplItem::Fn(mut method) = item {
            // Check if this method had the #[tool] attribute (replicate the check from Pass 1)
            let is_tool_method = method.attrs.iter().any(|attr| attr.path().is_ident("tool"));

            if is_tool_method {
                 // This is a tool function, remove #[doc] from parameters and #[tool] from the method
                 method.attrs.retain(|attr| !attr.path().is_ident("tool")); // Remove #[tool] attribute
                let mut modified_inputs = Punctuated::<FnArg, syn::token::Comma>::new();
                // Consume method.sig.inputs here
                for arg in method.sig.inputs.into_iter() {
                    if let FnArg::Typed(mut pat_type) = arg {
                        // Filter out #[doc] attributes
                        pat_type.attrs.retain(|attr| !attr.path().is_ident("doc"));
                        modified_inputs.push(FnArg::Typed(pat_type));
                    } else {
                        // Keep other argument types (like &self) as is
                        modified_inputs.push(arg);
                    }
                }
                method.sig.inputs = modified_inputs;
                final_items.push(ImplItem::Fn(method));
            } else {
                // Not a tool function, keep as is
                final_items.push(ImplItem::Fn(method));
            }
        } else {
            // Not a function, keep as is
            final_items.push(item);
        }
    }

    // Replace the original items with the modified ones for the final quote!
    item_impl.items = final_items;

    // Combine generated code, the ToolBox impl, and the modified original impl block
    let final_code = quote! {
        #generated_code

        #toolbox_impl

        #item_impl // Keep the original impl block, now modified
    };

    final_code.into()
}
