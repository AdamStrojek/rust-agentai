//! Tokio-backed runner and [`ToolBox`] for [`DeferredTask`]s.
//!
//! When a tool returns [`ToolReturn::Deferred`](crate::tool::ToolReturn::Deferred), the agent loop hands the unstarted
//! [`DeferredTask`] to the caller via [`AgentEvent::DeferredTask`](crate::agent::AgentEvent::DeferredTask)
//! (or [`AgentResult::unstarted_deferred_tasks`](crate::agent::AgentResult::unstarted_deferred_tasks))
//! and keeps a [`DeferredTaskHandle`](crate::tool::DeferredTaskHandle) to notify the model once
//! the task finishes.
//!
//! [`TokioDeferredTaskToolBox`] spawns those tasks onto the Tokio runtime and gives the model
//! tools to inspect and retrieve their results:
//! - `list_deferred_tasks`: lists known tasks with their ID, tool name and status.
//! - `get_deferred_task`: returns the completed [`ToolOutput`] (including any [`Media`](crate::Media)
//!   parts) for a finished task, or reports that it is still running.
//! - `wait_deferred_task`: waits up to a timeout for a task to finish without cancelling it if
//!   the timeout expires.

use crate::tool::{
    DeferredTask, DeferredTaskHandle, Tool, ToolBox, ToolError, ToolOutput, ToolResult, toolbox,
};
use log::warn;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use tokio::sync::Notify;

/// Default timeout used by `wait_deferred_task` when the model does not provide `timeout_seconds`.
const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

struct NotifyOnDrop(Arc<Notify>);

impl Drop for NotifyOnDrop {
    fn drop(&mut self) {
        self.0.notify_waiters();
    }
}

#[derive(Serialize)]
struct TaskSummary<'a> {
    id: &'a str,
    tool: &'a str,
    status: &'static str,
}

/// Spawns [`DeferredTask`]s on Tokio and exposes tools (`list_deferred_tasks`,
/// `get_deferred_task`, `wait_deferred_task`) so the model can check and fetch their results.
///
/// Cloning a `TokioDeferredTaskToolBox` shares the same underlying task table, so you can pass a
/// clone to [`Agent::with_toolbox`](crate::agent::Agent::with_toolbox) and keep another in your
/// event loop to call [`spawn`](Self::spawn).
#[derive(Clone)]
pub struct TokioDeferredTaskToolBox {
    tasks: Arc<Mutex<HashMap<String, DeferredTaskHandle>>>,
    notify: Arc<Notify>,
    default_timeout: Duration,
}

impl Default for TokioDeferredTaskToolBox {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TokioDeferredTaskToolBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokioDeferredTaskToolBox")
            .field("default_timeout", &self.default_timeout)
            .finish_non_exhaustive()
    }
}

impl TokioDeferredTaskToolBox {
    /// Creates an empty [`TokioDeferredTaskToolBox`].
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
            notify: Arc::new(Notify::new()),
            default_timeout: DEFAULT_WAIT_TIMEOUT,
        }
    }

    /// Sets the timeout used by `wait_deferred_task` when `timeout_seconds` is not given.
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Spawns `task` onto the Tokio runtime and tracks its [`DeferredTaskHandle`] under `task.id`.
    ///
    /// If a task with the same `id` is already tracked, `task` is dropped (marking its handle as
    /// cancelled) and not spawned.
    pub fn spawn(&self, task: DeferredTask) {
        {
            let mut tasks = self.tasks.lock().unwrap_or_else(PoisonError::into_inner);
            if tasks.contains_key(&task.id) {
                warn!(
                    "Duplicate deferred task id '{}' for tool '{}'; dropping duplicate task",
                    task.id, task.tool
                );
                return;
            }
            tasks.insert(task.id.clone(), task.handle());
        }
        let notify = NotifyOnDrop(self.notify.clone());
        tokio::spawn(async move {
            task.run().await;
            drop(notify);
        });
    }

    /// If `task_id` has finished, removes it from the table and returns its result.
    /// Returns `Ok(None)` when the task is still running, `Err(ToolError::Failed)` when it was
    /// cancelled or failed, or `Err(ToolError::InvalidArguments)` when `task_id` is unknown.
    fn try_take_finished(&self, task_id: &str) -> ToolResult<Option<ToolOutput>> {
        let mut tasks = self.tasks.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(handle) = tasks.get(task_id) else {
            return Err(ToolError::InvalidArguments(format!(
                "unknown deferred task '{task_id}'"
            )));
        };
        if !handle.is_finished() {
            return Ok(None);
        }
        let handle = tasks.remove(task_id).unwrap();
        if handle.is_cancelled() {
            return Err(ToolError::Failed(format!(
                "Deferred task '{task_id}' ({}) was cancelled before completing.",
                handle.tool
            )));
        }
        let Some(result) = handle.take_result() else {
            return Err(ToolError::InvalidArguments(format!(
                "deferred task '{task_id}' result has already been retrieved"
            )));
        };
        
        match result {
            Ok(output) => Ok(Some(output)),
            Err(err) => Err(ToolError::Failed(err.to_string())),
        }
    }
}

#[toolbox]
impl TokioDeferredTaskToolBox {
    /// Lists deferred background tasks with their ID, tool name and status (running, completed, failed, cancelled).
    #[tool]
    pub fn list_deferred_tasks(&self) -> ToolResult {
        let tasks = self.tasks.lock().unwrap_or_else(PoisonError::into_inner);
        let mut list: Vec<TaskSummary<'_>> = tasks
            .iter()
            .map(|(id, handle)| TaskSummary {
                id,
                tool: &handle.tool,
                status: handle.status_str(),
            })
            .collect();
        list.sort_by(|a, b| a.id.cmp(b.id));
        serde_json::to_string(&list).map_err(ToolError::fatal)
    }

    /// Returns the result of a finished deferred task by its ID, or reports that it is still running.
    #[tool]
    pub fn get_deferred_task(
        &self,
        /// ID of the deferred task to retrieve.
        task_id: String,
    ) -> ToolResult<ToolOutput> {
        match self.try_take_finished(&task_id)? {
            Some(output) => Ok(output),
            None => Ok(ToolOutput::from_text(format!(
                "Task '{task_id}' is still running."
            ))),
        }
    }

    /// Waits up to `timeout_seconds` for a deferred task to finish and returns its result. If the
    /// timeout expires, the task keeps running in the background.
    #[tool]
    pub async fn wait_deferred_task(
        &self,
        /// ID of the deferred task to wait for.
        task_id: String,
        /// Maximum number of seconds to wait before returning if the task is still running.
        timeout_seconds: Option<u64>,
    ) -> ToolResult<ToolOutput> {
        let timeout = timeout_seconds
            .map(Duration::from_secs)
            .unwrap_or(self.default_timeout);

        let wait_fut = async {
            loop {
                let notified = self.notify.notified();
                if let Some(output) = self.try_take_finished(&task_id)? {
                    return Ok(output);
                }
                notified.await;
            }
        };

        match tokio::time::timeout(timeout, wait_fut).await {
            Ok(result) => result,
            Err(_) => Ok(ToolOutput::from_text(format!(
                "Task '{task_id}' is still running."
            ))),
        }
    }
}
