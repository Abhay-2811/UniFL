use near_sdk::test_utils::VMContextBuilder;
use near_sdk::{testing_env, AccountId, VMContext};

use crate::FederatedToken;

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

// Add more tests here 