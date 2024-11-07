# UniFL: Decentralized Federated Learning with NEAR Protocol

UniFL is a decentralized federated learning platform that combines machine learning with blockchain technology to create a collaborative, incentivized learning environment. Contributors train models locally and receive tokens based on their contributions' impact on the global model.

## Why It Matters

Federated Learning solves several critical problems in modern AI:

- **Privacy**: Data never leaves the user's device
- **Collaboration**: Multiple parties can train AI models together without sharing sensitive data
- **Decentralization**: No single entity controls the model or data
- **Incentivization**: Contributors are rewarded for improving the model

## Technical Overview

The project consists of three main components:

1. **Federated Learning System**
   - Python-based server and client using gRPC
   - TensorFlow for model training
   - Currently supports MNIST dataset (expandable to other datasets)

2. **NEAR Smart Contract**
   - Handles token distribution
   - Tracks contributor scores
   - Automated weekly rewards
   - Built with Rust

3. **Web Interface**
   - Next.js dashboard
   - NEAR wallet integration
   - Real-time training statistics
   - Contributor leaderboard
