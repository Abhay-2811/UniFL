use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::collections::{LookupMap, UnorderedMap};
use near_sdk::json_types::U128;
use near_sdk::{env, near_bindgen, AccountId, Balance, PanicOnDefault, Promise};

const TOTAL_SUPPLY: Balance = 1_000_000_000_000_000_000_000_000_000; // 1 billion tokens with 24 decimals
const WEEKLY_DISTRIBUTION: Balance = 1_000_000_000_000_000_000_000_000; // 1 million tokens
const DISTRIBUTION_PERIOD: u64 = 7 * 24 * 60 * 60 * 1_000_000_000; // 7 days in nanoseconds

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct FederatedToken {
    pub owner_id: AccountId,
    pub total_supply: Balance,
    pub balances: LookupMap<AccountId, Balance>,
    pub contributors: UnorderedMap<AccountId, Contributor>,
    pub last_distribution_time: u64,
}

#[derive(BorshDeserialize, BorshSerialize)]
pub struct Contributor {
    contribution_count: u32,
    total_impact: f64,
    average_impact: f64,
    last_update: u64,
    pending_rewards: Balance,
}

#[near_bindgen]
impl FederatedToken {
    #[init]
    pub fn new(owner_id: AccountId) -> Self {
        assert!(!env::state_exists(), "Already initialized");
        let mut this = Self {
            owner_id,
            total_supply: TOTAL_SUPPLY,
            balances: LookupMap::new(b"b"),
            contributors: UnorderedMap::new(b"c"),
            last_distribution_time: env::block_timestamp(),
        };
        this
    }

    #[payable]
    pub fn update_contributor_score(&mut self, contributor: AccountId, impact: f64) {
        assert_eq!(env::predecessor_account_id(), self.owner_id, "Only owner can update scores");
        
        let mut contributor_data = self.contributors.get(&contributor).unwrap_or(Contributor {
            contribution_count: 0,
            total_impact: 0.0,
            average_impact: 0.0,
            last_update: env::block_timestamp(),
            pending_rewards: 0,
        });

        contributor_data.contribution_count += 1;
        contributor_data.total_impact += impact;
        contributor_data.average_impact = contributor_data.total_impact / contributor_data.contribution_count as f64;
        contributor_data.last_update = env::block_timestamp();

        self.contributors.insert(&contributor, &contributor_data);
    }

    pub fn distribute_tokens(&mut self) -> Promise {
        assert!(
            env::block_timestamp() >= self.last_distribution_time + DISTRIBUTION_PERIOD,
            "Too early for distribution"
        );

        let mut total_score: f64 = 0.0;
        let mut eligible_contributors = Vec::new();

        // Calculate total score
        for (account_id, contributor) in self.contributors.iter() {
            if contributor.contribution_count > 0 {
                let score = calculate_score(&contributor);
                total_score += score;
                eligible_contributors.push((account_id, score));
            }
        }

        // Distribute tokens
        for (account_id, score) in eligible_contributors {
            let token_amount = (WEEKLY_DISTRIBUTION as f64 * score / total_score) as u128;
            let contributor = self.contributors.get(&account_id).unwrap();
            
            // Update pending rewards
            let mut updated_contributor = contributor;
            updated_contributor.pending_rewards += token_amount;
            self.contributors.insert(&account_id, &updated_contributor);

            // Transfer tokens
            Promise::new(account_id).transfer(token_amount);
        }

        self.last_distribution_time = env::block_timestamp();
        Promise::new(env::current_account_id())
    }

    pub fn get_contributor_stats(&self, account_id: AccountId) -> Option<(u32, f64, f64, U128)> {
        self.contributors.get(&account_id).map(|c| (
            c.contribution_count,
            c.total_impact,
            c.average_impact,
            U128(c.pending_rewards)
        ))
    }

    pub fn get_next_distribution_time(&self) -> U128 {
        U128((self.last_distribution_time + DISTRIBUTION_PERIOD).into())
    }
}

fn calculate_score(contributor: &Contributor) -> f64 {
    let base_score = contributor.contribution_count as f64;
    let impact_multiplier = if contributor.average_impact > 0.0 {
        1.0 + contributor.average_impact
    } else if contributor.average_impact == 0.0 {
        1.0
    } else {
        0.5
    };
    base_score * impact_multiplier
}

#[cfg(test)]
mod tests {
    use super::*;
    use near_sdk::test_utils::VMContextBuilder;
    use near_sdk::{testing_env, VMContext};

    fn get_context(predecessor_account_id: AccountId) -> VMContext {
        VMContextBuilder::new()
            .predecessor_account_id(predecessor_account_id)
            .build()
    }

    #[test]
    fn test_new() {
        let context = get_context(AccountId::new_unchecked("owner".to_string()));
        testing_env!(context);
        let contract = FederatedToken::new(AccountId::new_unchecked("owner".to_string()));
        assert_eq!(contract.total_supply, TOTAL_SUPPLY);
    }
} 