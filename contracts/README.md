# Federated Learning Token (FLT) - Tokenomics & Distribution

## Overview
The Federated Learning Token (FLT) is designed to incentivize participation in the federated learning network by rewarding contributors based on their positive impact on the global model's performance.

## Token Specifications
- **Name**: Federated Learning Token
- **Symbol**: FLT
- **Decimals**: 18
- **Total Supply**: 1,000,000,000 FLT (1 billion)
- **Initial Distribution**: None (all tokens start in contract)

## Distribution Mechanics

### Weekly Distribution Pool
- 1,000,000 FLT distributed weekly
- Fixed distribution period: 7 days
- Total weeks of distribution: ~19 years (1 billion / 1 million per week)

### Contribution Scoring
Contributors are scored based on:
1. Number of contributions
2. Average impact on model accuracy
3. Total positive impact on model performance

### Distribution Formula
contributorTokens = (weeklyPool contributorScore) / totalScores


### Score Calculation
- Base score from contribution count
- Multiplier based on positive impact:
  - Improved accuracy: +1.0x to +2.0x multiplier
  - Neutral impact: 1.0x multiplier
  - Negative impact: 0.5x multiplier

## Distribution Rules

### Eligibility
- Must have made at least one contribution in the distribution period
- Contribution must have been validated by the network
- Wallet address must be properly registered

### Timing
- Distributions occur automatically every 7 days
- Unclaimed tokens remain in the contract
- No manual triggers required

### Impact Calculation
impact = (new_accuracy - previous_accuracy) 100

- Positive impact: Improvement in model accuracy
- Neutral impact: No change in accuracy (±0.1%)
- Negative impact: Decrease in accuracy

## Anti-Gaming Measures

### Quality Control
- Contributions that severely decrease model performance are discounted
- Rolling average impact calculation prevents gaming through sporadic high-impact contributions
- Minimum contribution threshold for eligibility

### Distribution Caps
- Maximum weekly tokens per contributor: 100,000 FLT (10% of weekly pool)
- Minimum weekly tokens per eligible contributor: 1,000 FLT (0.1% of weekly pool)

## Token Utility

### Governance Rights
- Voting power proportional to token holdings
- Ability to propose and vote on:
  - Model architecture changes
  - Distribution parameter adjustments
  - Network upgrades

### Staking Benefits
- Ability to stake tokens for enhanced rewards
- Staking multiplier: 1.1x to 1.5x based on lockup period
- Minimum stake: 10,000 FLT
- Lockup periods: 1, 3, 6, or 12 months

## Technical Implementation

### Smart Contract Architecture
- ERC20-compliant token contract
- Automated distribution mechanism
- Transparent scoring system
- Built-in governance functionality

### Security Features
- Time-locked distributions
- Multi-signature requirements for parameter changes
- Emergency pause functionality
- Regular security audits

## Future Developments

### Planned Features
1. Dynamic weekly pool size based on network activity
2. Enhanced governance mechanisms
3. Cross-chain compatibility
4. Liquidity mining programs

### Governance Parameters
The following parameters can be adjusted through governance:
- Weekly distribution amount
- Distribution period length
- Score calculation weights
- Minimum contribution thresholds