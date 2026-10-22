//! The kernel's built-in tool set.
//!
//! `registry` holds the generic contract (specs, dispatch, the extension table);
//! this module registers only the engine's conversation tools. Hosts assemble
//! their own capabilities on top of this set.

use super::registry::{ToolClass, ToolRegistry};

impl ToolRegistry {
    /// Register the generic conversation tools.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r.register(crate::tools::todo::spec());
        r.register(crate::tools::delegate::spec());
        r.register(crate::tools::delegate::answer_child_spec());
        r.register(crate::tools::ask::spec());
        r.register(crate::tools::link::spec());
        r.register(crate::tools::batch::spec());
        r
    }

    /// The generic tools a delegated child may draw from. The host decides
    /// which additional capabilities its children may hold. `ask_parent` is the
    /// one tool that exists only here — the parent has no one to ask.
    pub fn child_builtins() -> Self {
        let mut r = Self::with_builtins().filtered(|s| {
            !crate::tools::delegate::PARENT_ONLY_TOOLS.contains(&s.name)
                && s.class != ToolClass::HumanInteraction
        });
        r.register(crate::tools::delegate::ask_parent_spec());
        r
    }
}
