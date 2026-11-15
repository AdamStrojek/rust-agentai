//! Dollar spend limits for runs, with the `budget` feature.

use super::{AgentError, Usage};
use llm_budget::{AgentBudget, BudgetError, ChargeContext, ModelPrice, PriceTable};
use std::sync::Arc;

/// The budget of one run and the price of its model.
pub(super) struct RunBudget {
    budget: Arc<AgentBudget>,
    price: ModelPrice,
    /// Model name as found in the price table, recorded with every charge
    model: String,
}

impl RunBudget {
    /// Looks up the price of `model` in `prices`, or in the table built into `llm-budget`.
    ///
    /// A model without a price is an error: charging it nothing would let the run pass the limit.
    pub(super) fn new(
        budget: Arc<AgentBudget>,
        prices: Option<&PriceTable>,
        model: &str,
    ) -> Result<Self, AgentError> {
        // The `provider::` prefix of genai model names is not part of the price table names
        let model = model.rsplit_once("::").map_or(model, |(_, name)| name);
        let prices = prices.unwrap_or_else(|| PriceTable::builtin());
        let price = prices.require(model).map_err(AgentError::Budget)?.clone();
        Ok(Self {
            budget,
            price,
            model: model.to_string(),
        })
    }

    /// Refuses the next model call when the kill switch is on or nothing is left to spend.
    pub(super) fn check(&self) -> Result<(), AgentError> {
        // `reserve` reports the kill switch and a cap that is already crossed
        self.budget
            .reserve(0.0)
            .map(drop)
            .map_err(AgentError::Budget)?;
        if self.budget.is_exhausted() {
            return Err(AgentError::Budget(BudgetError::AgentExhausted {
                agent_id: self.budget.agent_id.clone(),
                spent: self.budget.spent_usd(),
                limit: self.budget.limit_usd(),
            }));
        }
        Ok(())
    }

    /// Books the cost of one model call, even past the limit (the call has been made), and
    /// returns it in USD.
    pub(super) fn charge(&self, usage: &Usage) -> f64 {
        let usage = llm_budget::Usage {
            input_tokens: usage.input_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            cache_write_tokens: usage.cache_write_tokens,
            output_tokens: usage.output_tokens,
        };
        let context = ChargeContext::new()
            .model(self.model.clone())
            .tokens(usage.input_tokens, usage.output_tokens);
        self.budget
            .charge_usd_force(self.price.cost_usage(&usage), context)
            .recorded_usd
    }
}
