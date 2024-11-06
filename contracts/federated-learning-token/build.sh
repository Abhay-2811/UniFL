#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add wasm32 target if not already added
rustup target add wasm32-unknown-unknown

# Create res directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/res"

# Navigate to the workspace root (contracts directory)
cd "${SCRIPT_DIR}/.."

# Build the contract
RUSTFLAGS='-C link-arg=-s' cargo build --target wasm32-unknown-unknown --release

# Copy the wasm file to res directory in the contract folder
cp "target/wasm32-unknown-unknown/release/federated_learning_token.wasm" "${SCRIPT_DIR}/res/"