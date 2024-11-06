use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::collections::{LookupMap, UnorderedMap, LazyOption};
use near_sdk::json_types::U128;
use near_sdk::{env, near_bindgen, AccountId, PanicOnDefault, Promise};
use near_contract_standards::fungible_token::{
    metadata::{FungibleTokenMetadata, FungibleTokenMetadataProvider, FT_METADATA_SPEC},
    Balance,
};

use crate::utils::calculate_score;

const TOTAL_SUPPLY: Balance = 1_000_000_000_000_000_000_000_000_000; // 1 billion tokens with 24 decimals
const WEEKLY_DISTRIBUTION: Balance = 1_000_000_000_000_000_000_000_000; // 1 million tokens
const DISTRIBUTION_PERIOD: u64 = 5 * 60 * 1_000_000_000; // 5 minutes in nanoseconds

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct FederatedToken {
    pub owner_id: AccountId,
    pub total_supply: Balance,
    pub balances: LookupMap<AccountId, Balance>,
    pub contributors: UnorderedMap<AccountId, Contributor>,
    pub last_distribution_time: u64,
    pub metadata: LazyOption<FungibleTokenMetadata>,
}

#[derive(BorshDeserialize, BorshSerialize)]
pub struct Contributor {
    pub contribution_count: u32,
    pub total_impact: f64,
    pub average_impact: f64,
    pub last_update: u64,
    pub pending_rewards: Balance,
}

#[near_bindgen]
impl FederatedToken {
    #[init]
    pub fn new(owner_id: AccountId) -> Self {
        assert!(!env::state_exists(), "Already initialized");
        let mut this = Self {
            owner_id: owner_id.clone(),
            total_supply: TOTAL_SUPPLY,
            balances: LookupMap::new(b"b"),
            contributors: UnorderedMap::new(b"c"),
            last_distribution_time: env::block_timestamp(),
            metadata: LazyOption::new(
                b"m",
                Some(&FungibleTokenMetadata {
                    spec: FT_METADATA_SPEC.to_string(),
                    name: "Federated Learning Token".to_string(),
                    symbol: "FLT".to_string(),
                    icon: None,
                    reference: None,
                    reference_hash: None,
                    decimals: 24,
                }),
            ),
        };
        // Initialize owner's balance with total supply
        this.balances.insert(&owner_id, &TOTAL_SUPPLY);
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

    #[payable]
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
            
            // Update balances
            let current_balance = self.balances.get(&account_id).unwrap_or(0);
            self.balances.insert(&account_id, &(current_balance + token_amount));
            
            // Update contributor's pending rewards
            if let Some(mut contributor) = self.contributors.get(&account_id) {
                contributor.pending_rewards += token_amount;
                self.contributors.insert(&account_id, &contributor);
            }

            // Log the distribution
            env::log_str(&format!(
                "Distributed {} tokens to {}",
                token_amount, account_id
            ));
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

    pub fn get_balance(&self, account_id: AccountId) -> U128 {
        U128(self.balances.get(&account_id).unwrap_or(0))
    }

    pub fn get_total_supply(&self) -> U128 {
        U128(self.total_supply)
    }

    #[payable]
    pub fn transfer(&mut self, receiver_id: AccountId, amount: U128) {
        let sender_id = env::predecessor_account_id();
        let amount: Balance = amount.into();

        let sender_balance = self.balances.get(&sender_id).expect("Sender not found");
        assert!(sender_balance >= amount, "Not enough balance");

        // Update sender balance
        self.balances.insert(&sender_id, &(sender_balance - amount));

        // Update receiver balance
        let receiver_balance = self.balances.get(&receiver_id).unwrap_or(0);
        self.balances.insert(&receiver_id, &(receiver_balance + amount));

        // Log the transfer
        env::log_str(&format!(
            "Transfer {} tokens from {} to {}",
            amount, sender_id, receiver_id
        ));
    }

    pub fn get_owner_id(&self) -> AccountId {
        self.owner_id.clone()
    }
}

#[near_bindgen]
impl FungibleTokenMetadataProvider for FederatedToken {
    fn ft_metadata(&self) -> FungibleTokenMetadata {
        self.metadata.get().unwrap()
    }
}
