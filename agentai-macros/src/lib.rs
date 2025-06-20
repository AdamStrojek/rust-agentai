use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, Error, Expr, FnArg, Ident, ImplItem, ItemImpl, Lit, Meta, MetaNameValue, Pat
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
