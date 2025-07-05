#!/bin/bash
#
# Description:
#   This script runs all Rust examples found in the `examples` directory.
#   It logs the output of each example to a file in the `logs` directory
#   and exits with a non-zero status code if any example fails.
#
# Environment Variables:
#   - AGENTAI_BASE_URL: The base URL for the AI model's API.
#   - AGENTAI_API_KEY: The API key for authentication.
#   - AGENTAI_MODEL: The identifier for the model to be used.
#
# Usage:
#   export AGENTAI_BASE_URL="your_base_url"
#   export AGENTAI_API_KEY="your_api_key"
#   export AGENTAI_MODEL="your_model"
#   ./scripts/run-examples.sh

# Ensure that the exit code of a pipeline is the exit code of the last command to exit with a non-zero status.
set -o pipefail

# --- Configuration ---
LOG_DIR="logs"
FINAL_EXIT_CODE=0

# --- Pre-flight Checks ---
if [[ -z "$AGENTAI_BASE_URL" || -z "$AGENTAI_API_KEY" || -z "$AGENTAI_MODEL" ]]; then
  echo "Error: Please set the required environment variables: AGENTAI_BASE_URL, AGENTAI_API_KEY, AGENTAI_MODEL" >&2
  exit 1
fi

# --- Main Execution ---
mkdir -p "$LOG_DIR"

echo "Running examples with model: $AGENTAI_MODEL"
echo "Log directory: $LOG_DIR"
echo ""

for example_file in examples/*.rs; do
  EXAMPLE_NAME=$(basename "$example_file" .rs)
  LOG_FILE="$LOG_DIR/${EXAMPLE_NAME}.log"

  echo "--- Running example: $EXAMPLE_NAME ---"

  # Execute the example, stream its output to the console, and save it to a log file.
  # The exit code of `cargo run` is captured correctly due to `set -o pipefail`.
  if cargo run --release --example "$EXAMPLE_NAME" 2>&1 | tee "$LOG_FILE"; then
    echo "Example '$EXAMPLE_NAME' finished successfully."
  else
    CARGO_EXIT_CODE=$?
    # The '::error::' prefix is for GitHub Actions to highlight the line as an error.
    echo "::error::Example '$EXAMPLE_NAME' failed with exit code $CARGO_EXIT_CODE. See log: $LOG_FILE" >&2
    FINAL_EXIT_CODE=1
  fi
  echo "-------------------------------------"
  echo ""
done

# --- Final Status ---
if [ $FINAL_EXIT_CODE -ne 0 ]; then
  echo "::error::One or more examples failed. Please check the logs." >&2
fi

exit $FINAL_EXIT_CODE
