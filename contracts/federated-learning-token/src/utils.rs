use crate::token::Contributor;

pub(crate) fn calculate_score(contributor: &Contributor) -> f64 {
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