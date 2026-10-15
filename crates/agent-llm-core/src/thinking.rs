//! Thinking-level vocabulary and selection rules (pure data, no config I/O).

use std::collections::HashMap;

/// The pi thinking levels, in UI order — the only valid `thinkingLevelMap` keys.
pub const THINKING_LEVELS: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/// A model's explicit map is the selectable-level allowlist. `off` is always
/// available when a map exists; missing/null/empty entries expose no tier.
pub fn available_thinking_levels(map: Option<&HashMap<String, Option<String>>>) -> Vec<String> {
    let Some(map) = map else { return Vec::new() };
    THINKING_LEVELS
        .iter()
        .filter(|&&level| {
            level == "off"
                || map
                    .get(level)
                    .and_then(|value| value.as_deref())
                    .is_some_and(|value| !value.trim().is_empty())
        })
        .map(|level| (*level).to_string())
        .collect()
}

/// Preserve a supported selection (including explicit off). On restore or
/// model switch, stale tiers fall back to medium, the first supported on
/// tier, then off. No declared map means the provider's default, not a tier
/// inherited from a different model.
pub fn normalize_thinking_level(
    requested: Option<&str>,
    map: Option<&HashMap<String, Option<String>>>,
) -> Option<String> {
    let levels = available_thinking_levels(map);
    let requested = requested.map(|level| level.trim().to_ascii_lowercase());
    requested
        .filter(|level| levels.contains(level))
        .or_else(|| {
            levels
                .iter()
                .find(|level| level.as_str() == "medium")
                .cloned()
        })
        .or_else(|| levels.iter().find(|level| level.as_str() != "off").cloned())
        .or_else(|| levels.first().cloned())
}
