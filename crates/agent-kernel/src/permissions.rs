//! `PermissionGate` — policy between "model asked for a tool" and "tool runs".
//!
//! | Mode                | Behavior                                              |
//! |---------------------|-------------------------------------------------------|
//! | `default`           | ask for non-readonly tools; readonly auto-run         |
//! | `acceptEdits`       | workspace mutations auto-approve; other writes ask   |
//! | `auto`              | auto-approve what passes safety checks; escalate rest |
//! | `dontAsk`           | only pre-approved tools + readonly calls             |
//! | `bypassPermissions` | approve all except `deny` rules + destructive `ask`   |
//!
//! Precedence: `deny > ask > allow`, fed by CLI flags and `config.toml`. The
//! gate is synchronous policy, not a channel: the engine calls `decide()` and on
//! `Ask` pauses in `AwaitingToolConfirmation` until a `ToolDecision` arrives.

use std::collections::HashSet;

use crate::tools::registry::ToolClass;
pub use agent_tool::CallRisk;

/// The five modes — `from_str` accepts the labels the prompt/config use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Default,
    AcceptEdits,
    Auto,
    DontAsk,
    Bypass,
}

impl PermissionMode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "acceptEdits" | "accept-edits" => Self::AcceptEdits,
            "auto" => Self::Auto,
            "dontAsk" | "dont-ask" => Self::DontAsk,
            "bypassPermissions" | "bypass" => Self::Bypass,
            _ => Self::Default,
        }
    }

    /// Canonical wire label — the inverse of `from_str`, used to report a
    /// live gate's mode back to the UI (`model_info`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::Auto => "auto",
            Self::DontAsk => "dontAsk",
            Self::Bypass => "bypassPermissions",
        }
    }

    /// The Chinese name the UI shows for this mode — the same wording as the
    /// composer's permission picker, so a switch notice reads like the menu.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Default => "默认",
            Self::AcceptEdits => "接受编辑",
            Self::Auto => "自动",
            Self::DontAsk => "不询问",
            Self::Bypass => "跳过权限",
        }
    }
}

/// What the gate returns for one tool call.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Run it — no pause.
    Allow,
    /// Refuse outright (a `deny` rule, or `dontAsk` with no pre-approval).
    Deny { reason: String },
    /// Pause in `AwaitingToolConfirmation`; UI decides.
    Ask { diff_summary: String },
}

/// A tool-specific permission policy, registered on the gate by tool name. The
/// gate knows no tool: a host that ships a high-risk tool (a desktop driver, a
/// remote shell) registers the rules that tool needs, and the gate just asks.
///
/// All three hooks default to "no opinion", so a policy only states what makes
/// its tool special.
pub trait ToolPolicy: Send + Sync {
    /// Ask by default, above the permission mode: even `auto` and `bypass` ask
    /// until the session whitelist (or an explicit allow rule) grants it.
    /// `dontAsk` keeps its "no questions" contract and denies instead.
    fn ask_by_default(&self) -> bool {
        false
    }

    /// Per-action escalation applied on top of the gate's decision. `action`
    /// is the tool's own sub-action (see `agent_tool::Review::action`); `allow`
    /// is the session's allow set. The gate preserves an existing Ask/Deny
    /// before calling this hook and enforces `can_ask` on its result.
    fn escalate(
        &self,
        _action: &str,
        decision: Decision,
        _allow: &HashSet<String>,
        _can_ask: bool,
        _diff_summary: &str,
    ) -> Decision {
        decision
    }

    /// Allow-rule names this policy owns, dropped by [`PermissionGate::revoke_grants`].
    fn grants(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Rule sets — `deny` always wins, then `ask`, then `allow`.
#[derive(Debug, Default, Clone)]
pub struct PermissionRules {
    /// Tool names that never run (e.g. `"bash"`, `"fuzzy_patch"`).
    pub deny: HashSet<String>,
    /// Tool names that always pause for confirmation.
    pub ask: HashSet<String>,
    /// Tool names pre-approved (skip the pause in `default`/`dontAsk`).
    pub allow: HashSet<String>,
}

pub struct PermissionGate {
    mode: PermissionMode,
    rules: PermissionRules,
    /// Tool-specific policies, keyed by tool name.
    policies: std::collections::HashMap<String, std::sync::Arc<dyn ToolPolicy>>,
    /// Subagent context: nothing answers `Ask`, so escalations deny
    /// outright and the refusal is reported back to the child model.
    headless: bool,
    /// Names the user granted with "本会话允许" — kept apart from `rules.allow`
    /// because a delegated child inherits the configured rules but not a click
    /// that covered one turn.
    session_grants: std::collections::HashSet<String>,
}

impl PermissionGate {
    pub fn new(mode: PermissionMode, rules: PermissionRules) -> Self {
        Self {
            mode,
            rules,
            policies: Default::default(),
            headless: false,
            session_grants: Default::default(),
        }
    }

    /// Gate for a delegated subagent — `auto` baseline (readonly, edits,
    /// and safe shell auto-run) with `headless` on: anything that would
    /// pause for human approval denies instead, since no UI answers it.
    pub fn for_subagent() -> Self {
        Self {
            mode: PermissionMode::Auto,
            rules: PermissionRules::default(),
            policies: Default::default(),
            headless: true,
            session_grants: Default::default(),
        }
    }

    /// Gate for a delegated child: the parent's own mode and rules, so a child
    /// is bounded exactly like the turn that spawned it instead of by a
    /// separate policy nobody chose. Two differences, both deliberate:
    ///
    /// - The user's "本会话允许" clicks stay with the parent. They answered one
    ///   task, and a child is a different one.
    /// - `bypass` drops to `auto`. A child works unwatched; an unconditional
    ///   yes is the one mode that must not travel with it.
    ///
    /// `can_ask` says whether an escalation can still reach the user: when it
    /// can, a child pauses on a real card instead of denying.
    pub fn child_of(parent: &PermissionGate, can_ask: bool) -> Self {
        let mut rules = parent.rules.clone();
        for granted in &parent.session_grants {
            rules.allow.remove(granted);
        }
        Self {
            mode: if parent.mode == PermissionMode::Bypass {
                PermissionMode::Auto
            } else {
                parent.mode
            },
            rules,
            policies: parent.policies.clone(),
            headless: !can_ask,
            session_grants: Default::default(),
        }
    }

    pub fn from_mode_str(mode: &str) -> Self {
        Self::new(PermissionMode::from_str(mode), PermissionRules::default())
    }

    pub fn mode(&self) -> PermissionMode {
        self.mode
    }

    /// Merge repo-local rules (`.agent/config.toml` → deny/ask/allow lists).
    pub fn merge_rules(&mut self, rules: PermissionRules) {
        self.rules.deny.extend(rules.deny);
        self.rules.ask.extend(rules.ask);
        self.rules.allow.extend(rules.allow);
    }

    /// Switch the permission mode in place, keeping the session's rules. A
    /// mode change must not silently drop a "本会话允许" grant the user gave
    /// (the overlay is still up and says the agent is allowed to drive).
    pub fn set_mode(&mut self, mode: PermissionMode) {
        self.mode = mode;
    }

    /// Register a tool-specific policy. The gate applies it to every call of
    /// `tool_name`; a host with no such tool simply registers nothing.
    pub fn with_policy(
        mut self,
        tool_name: impl Into<String>,
        policy: std::sync::Arc<dyn ToolPolicy>,
    ) -> Self {
        self.policies.insert(tool_name.into(), policy);
        self
    }

    /// Register a policy on an existing gate (the engine's shared one).
    pub fn set_policy(
        &mut self,
        tool_name: impl Into<String>,
        policy: std::sync::Arc<dyn ToolPolicy>,
    ) {
        self.policies.insert(tool_name.into(), policy);
    }

    /// Per-action escalation from the tool's own policy, applied on top of
    /// [`decide`]. A tool with no policy, or an action the policy has no
    /// opinion on, passes through unchanged. Under `dontAsk` (and in a
    /// headless subagent) nobody can answer, so a policy is told it cannot
    /// ask and should deny instead.
    pub fn escalate(
        &self,
        tool_name: &str,
        action: &str,
        decision: Decision,
        diff_summary: impl Into<String>,
    ) -> Decision {
        if !matches!(decision, Decision::Allow) {
            return decision;
        }
        let Some(policy) = self.policies.get(tool_name) else {
            return decision;
        };
        let can_ask = self.mode != PermissionMode::DontAsk && !self.headless;
        let decision = policy.escalate(
            action,
            decision,
            &self.rules.allow,
            can_ask,
            &diff_summary.into(),
        );
        match decision {
            Decision::Ask { .. } if !can_ask => Decision::Deny {
                reason: format!(
                    "`{tool_name}` requires human approval — unavailable in this context"
                ),
            },
            decision => decision,
        }
    }

    /// Whitelist `tool_name` for the rest of this session. The grant lives on
    /// this gate, dies with the actor, and never changes persistent config.
    pub fn allow_tool_for_session(&mut self, tool_name: &str) {
        self.rules.allow.insert(tool_name.to_string());
        self.session_grants.insert(tool_name.to_string());
    }

    /// Drop every grant a tool has: the tool itself and any allow rule its
    /// policy owns. Returns whether the tool was granted, so the caller knows
    /// whether a "grant ended" notice is due. Run at the end of each turn: a
    /// grant covers the task it was given for, not whatever the session is
    /// asked next.
    pub fn revoke_grants(&mut self, tool_name: &str) -> bool {
        let was = self.rules.allow.remove(tool_name);
        self.session_grants.remove(tool_name);
        if let Some(policy) = self.policies.get(tool_name) {
            for rule in policy.grants() {
                self.rules.allow.remove(&rule);
            }
        }
        was
    }

    /// Remove `tool_name` from the session whitelist — the overlay's
    /// "停止" button path. Idempotent: removing a tool that was never
    /// whitelisted is a no-op, which is the right answer for a revoke
    /// arriving after the actor dropped the entry some other way.
    pub fn revoke_tool_for_session(&mut self, tool_name: &str) {
        self.rules.allow.remove(tool_name);
        self.session_grants.remove(tool_name);
    }

    /// Decide one tool call. `tool_name` + `is_readonly` come from the
    /// registry; `risk` is the tool's assessment of this call.
    /// `class` is the registry's execution class (`None` for an unknown spec);
    /// `diff_summary` is pre-computed for `Ask` cards.
    ///
    /// Precedence: deny > ask > allow > mode default.
    pub fn decide(
        &self,
        tool_name: &str,
        class: Option<ToolClass>,
        is_readonly: bool,
        risk: CallRisk,
        diff_summary: impl Into<String>,
    ) -> Decision {
        let d = self.decide_inner(tool_name, class, is_readonly, risk, diff_summary);
        if self.headless {
            match d {
                Decision::Ask { .. } => Decision::Deny {
                    reason: "requires human approval — unavailable inside a delegated subagent"
                        .into(),
                },
                d => d,
            }
        } else {
            d
        }
    }

    fn decide_inner(
        &self,
        tool_name: &str,
        class: Option<ToolClass>,
        is_readonly: bool,
        risk: CallRisk,
        diff_summary: impl Into<String>,
    ) -> Decision {
        // 1. Explicit deny — absolute.
        if self.rules.deny.contains(tool_name) {
            return Decision::Deny {
                reason: format!("`{tool_name}` denied by rule"),
            };
        }
        // A call's risk can require approval above the mode default.
        if risk == CallRisk::RequiresApproval && !self.rules.allow.contains(tool_name) {
            return Decision::Ask {
                diff_summary: diff_summary.into(),
            };
        }
        // A host policy can require approval above the mode default.
        // Explicit deny still wins; an allow grant skips this requirement.
        if self
            .policies
            .get(tool_name)
            .is_some_and(|p| p.ask_by_default())
            && !self.rules.allow.contains(tool_name)
        {
            if self.mode == PermissionMode::DontAsk {
                return Decision::Deny {
                    reason: format!("`{tool_name}` requires human approval"),
                };
            }
            return Decision::Ask {
                diff_summary: diff_summary.into(),
            };
        }
        // 2. Explicit ask.
        if self.rules.ask.contains(tool_name) {
            return Decision::Ask {
                diff_summary: diff_summary.into(),
            };
        }
        // 3. Explicit allow.
        if self.rules.allow.contains(tool_name) {
            return Decision::Allow;
        }
        let is_readonly = is_readonly || risk == CallRisk::Readonly;
        // 4. Mode default.
        match self.mode {
            PermissionMode::Bypass => Decision::Allow,
            PermissionMode::Default => {
                if is_readonly {
                    Decision::Allow
                } else {
                    Decision::Ask {
                        diff_summary: diff_summary.into(),
                    }
                }
            }
            PermissionMode::AcceptEdits => {
                if is_readonly || class == Some(ToolClass::WorkspaceMutation) {
                    Decision::Allow
                } else {
                    Decision::Ask {
                        diff_summary: diff_summary.into(),
                    }
                }
            }
            PermissionMode::Auto => {
                // Calls requiring approval were already escalated above.
                Decision::Allow
            }
            PermissionMode::DontAsk => {
                if is_readonly {
                    Decision::Allow
                } else {
                    Decision::Deny {
                        reason: "dontAsk: tool not pre-approved".into(),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_is_chinese_and_follows_aliases() {
        // The switch notice reads the label of the parsed mode, so aliases
        // (`bypass`, `dont-ask`) must land on the same Chinese name.
        assert_eq!(PermissionMode::from_str("dontAsk").label(), "不询问");
        assert_eq!(PermissionMode::from_str("dont-ask").label(), "不询问");
        assert_eq!(PermissionMode::from_str("acceptEdits").label(), "接受编辑");
        assert_eq!(PermissionMode::from_str("bypass").label(), "跳过权限");
        assert_eq!(PermissionMode::from_str("auto").label(), "自动");
        assert_eq!(PermissionMode::from_str("default").label(), "默认");
    }

    fn gate(mode: &str) -> PermissionGate {
        PermissionGate::from_mode_str(mode)
    }

    /// A child answers to the parent's rules, so a project that denies `bash`
    /// still denies it one level down — the old fixed `auto` baseline quietly
    /// allowed what the session had forbidden.
    #[test]
    fn a_child_inherits_the_parents_mode_and_rules() {
        let mut rules = PermissionRules::default();
        rules.deny.insert("bash".into());
        let parent = PermissionGate::new(PermissionMode::Default, rules);
        let child = PermissionGate::child_of(&parent, true);
        assert!(matches!(
            child.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Deny { .. }
        ));
        // The mode travels too: `default` asks for a write in the child just
        // as it does in the parent.
        assert!(matches!(
            child.decide("fuzzy_patch", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
    }

    /// "本会话允许" answered one task. A child is a different task, so the
    /// click does not travel with it — but a rule the project configured does.
    #[test]
    fn a_child_does_not_inherit_a_click_but_does_inherit_a_configured_allow() {
        let mut parent = PermissionGate::new(PermissionMode::Default, PermissionRules::default());
        parent.allow_tool_for_session("bash");
        assert_eq!(
            parent.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        );
        let child = PermissionGate::child_of(&parent, true);
        assert!(matches!(
            child.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));

        let mut rules = PermissionRules::default();
        rules.allow.insert("bash".into());
        let configured = PermissionGate::new(PermissionMode::Default, rules);
        let child = PermissionGate::child_of(&configured, true);
        assert_eq!(
            child.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        );
    }

    /// A child pauses on a question only when the answer can reach it; a
    /// headless run denies instead of parking on a card nobody will see.
    #[test]
    fn a_child_asks_only_when_the_user_can_answer() {
        let parent = PermissionGate::new(PermissionMode::Default, PermissionRules::default());
        assert!(matches!(
            PermissionGate::child_of(&parent, true).decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
        assert!(matches!(
            PermissionGate::child_of(&parent, false).decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Deny { .. }
        ));
    }

    struct AskPolicy;

    impl ToolPolicy for AskPolicy {
        fn ask_by_default(&self) -> bool {
            true
        }

        fn escalate(
            &self,
            _: &str,
            _: Decision,
            _: &HashSet<String>,
            _: bool,
            summary: &str,
        ) -> Decision {
            Decision::Ask {
                diff_summary: summary.into(),
            }
        }

        fn grants(&self) -> Vec<String> {
            vec!["guarded:extra".into()]
        }
    }

    fn guarded_gate(mode: &str) -> PermissionGate {
        gate(mode).with_policy("guarded", std::sync::Arc::new(AskPolicy))
    }

    #[test]
    fn default_asks_for_writes_not_reads() {
        let g = gate("default");
        assert!(matches!(
            g.decide("smart_read", None, true, CallRisk::Normal, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("fuzzy_patch", None, false, CallRisk::Normal, "engine.rs +5"),
            Decision::Ask { .. }
        ));
    }

    #[test]
    fn default_allows_readonly_shell_but_asks_writes() {
        let g = gate("default");
        // readonly shell verbs auto-run — ls/cat/git status/cargo test…
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Readonly, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Readonly, ""),
            Decision::Allow
        ));
        // …but writes, installs, builds, and chains still ask.
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::RequiresApproval, ""),
            Decision::Ask { .. }
        ));
    }

    #[test]
    fn default_allows_todo_without_asking() {
        let g = gate("default");
        assert!(matches!(
            g.decide("todo", None, true, CallRisk::Normal, ""),
            Decision::Allow
        ));
    }

    #[test]
    fn deny_beats_everything() {
        let mut g = gate("bypassPermissions");
        g.merge_rules(PermissionRules {
            deny: ["bash".into()].into_iter().collect(),
            ..Default::default()
        });
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Readonly, ""),
            Decision::Deny { .. }
        ));
    }

    #[test]
    fn destructive_shell_asks_even_in_bypass() {
        let g = gate("bypassPermissions");
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::RequiresApproval, ""),
            Decision::Ask { .. }
        ));
    }

    #[test]
    fn accept_edits_skips_patch_asks_shell() {
        let g = gate("acceptEdits");
        assert!(matches!(
            g.decide(
                "fuzzy_patch",
                Some(ToolClass::WorkspaceMutation),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Allow
        ));
        // non-readonly shell (an install) still asks even in acceptEdits.
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
    }

    /// The class is the contract: workspace mutations are auto-approved in
    /// `acceptEdits`; an absent or different class does not widen the gate.
    #[test]
    fn accept_edits_trusts_the_tool_class() {
        let g = gate("acceptEdits");
        assert!(matches!(
            g.decide(
                "brand_new_writer",
                Some(ToolClass::WorkspaceMutation),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Allow
        ));
        // A familiar-looking name cannot replace declared metadata.
        assert!(matches!(
            g.decide("fs_patch", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
        // A classed non-writer still asks — metadata must not widen the gate.
        assert!(matches!(
            g.decide(
                "some_process",
                Some(ToolClass::Process),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Ask { .. }
        ));
    }

    #[test]
    fn a_host_policy_asks_above_the_mode_default() {
        for mode in ["default", "acceptEdits", "auto", "bypassPermissions"] {
            let g = guarded_gate(mode);
            assert!(
                matches!(
                    g.decide(
                        "guarded",
                        Some(ToolClass::Process),
                        false,
                        CallRisk::Normal,
                        ""
                    ),
                    Decision::Ask { .. }
                ),
                "mode {mode} should honour the policy"
            );
        }

        // The host policy applies even if the tool is marked readonly.
        let g = guarded_gate("dontAsk");
        assert!(matches!(
            g.decide(
                "guarded",
                Some(ToolClass::Observation),
                true,
                CallRisk::Normal,
                ""
            ),
            Decision::Deny { .. }
        ));

        for mode in [
            "default",
            "auto",
            "bypassPermissions",
            "acceptEdits",
            "dontAsk",
        ] {
            let mut g = guarded_gate(mode);
            g.allow_tool_for_session("guarded");
            assert!(
                matches!(
                    g.decide(
                        "guarded",
                        Some(ToolClass::Process),
                        false,
                        CallRisk::Normal,
                        ""
                    ),
                    Decision::Allow
                ),
                "session-allowed tool should run in {mode}"
            );
        }
    }

    #[test]
    fn an_unregistered_tool_has_only_the_generic_rules() {
        assert_eq!(
            gate("auto").decide(
                "guarded",
                Some(ToolClass::Process),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Allow
        );
    }

    #[test]
    fn dont_ask_allows_readonly_shell_denies_write() {
        let g = gate("dontAsk");
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Readonly, ""),
            Decision::Allow
        ));
        // a build writes artifacts → not readonly → denied in dontAsk.
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Deny { .. }
        ));
    }

    #[test]
    fn auto_allows_safe_shell_and_edits_asks_destructive() {
        let g = gate("auto");
        // file edits auto-run in auto
        assert!(matches!(
            g.decide("fuzzy_patch", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("apply_patch", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("write_file", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));

        // safe non-readonly shell commands auto-run in auto
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));

        // destructive shell commands escalate to Ask even in auto
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::RequiresApproval, ""),
            Decision::Ask { .. }
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::RequiresApproval, ""),
            Decision::Ask { .. }
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::RequiresApproval, ""),
            Decision::Ask { .. }
        ));
    }

    #[test]
    fn accept_edits_allows_readonly_shell() {
        let g = gate("acceptEdits");
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Readonly, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Readonly, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Readonly, ""),
            Decision::Allow
        ));
    }

    /// `auto` does not flatten rule precedence: an explicit `ask` still
    /// pauses on a tool the mode would otherwise run, and `deny` still
    /// wins outright. The mode only defaults the *ungoverned* case.
    #[test]
    fn auto_keeps_rule_precedence() {
        let mut g = gate("auto");
        g.merge_rules(PermissionRules {
            ask: ["fuzzy_patch".into()].into_iter().collect(),
            deny: ["computer".into()].into_iter().collect(),
            ..Default::default()
        });
        // ask > auto-allow
        assert!(matches!(
            g.decide("fuzzy_patch", None, false, CallRisk::Normal, ""),
            Decision::Ask { .. }
        ));
        // deny > every interactive surface (computer would Ask, deny is stronger)
        assert!(matches!(
            g.decide(
                "computer",
                Some(ToolClass::Process),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Deny { .. }
        ));
        // ungoverned calls still auto-run
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));
    }

    /// An `allow` rule on `bash` is the documented escape hatch: destructive
    /// commands then skip the Ask even in `auto`. This locks the contract —
    /// if it ever changes, this test is where the conversation must happen.
    #[test]
    fn allow_rule_overrides_destructive_escalation() {
        let mut g = gate("auto");
        g.allow_tool_for_session("bash");
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::RequiresApproval, ""),
            Decision::Allow
        ));
    }

    /// Delegated subagents run `auto` + `headless`: everything `auto` would
    /// still Ask for becomes Deny — nobody
    /// is there to answer. This is the subagent's whole safety story.
    #[test]
    fn headless_auto_denies_what_would_ask() {
        let mut g = PermissionGate::for_subagent();
        g.set_policy("guarded", std::sync::Arc::new(AskPolicy));
        // Safe calls still run.
        assert!(matches!(
            g.decide("smart_read", None, true, CallRisk::Normal, ""),
            Decision::Allow
        ));
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::Normal, ""),
            Decision::Allow
        ));
        // Anything interactive-mode-asked denies.
        assert!(matches!(
            g.decide("bash", None, false, CallRisk::RequiresApproval, ""),
            Decision::Deny { .. }
        ));
        assert!(matches!(
            g.decide(
                "guarded",
                Some(ToolClass::Process),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Deny { .. }
        ));
    }

    #[test]
    fn a_policy_cannot_ask_where_nobody_can_answer() {
        for mut g in [
            guarded_gate("dontAsk"),
            PermissionGate::for_subagent().with_policy("guarded", std::sync::Arc::new(AskPolicy)),
        ] {
            g.allow_tool_for_session("guarded");
            assert!(matches!(
                g.escalate("guarded", "action", Decision::Allow, "x"),
                Decision::Deny { .. }
            ));
        }
    }

    /// Escalation never softens: an Ask/Deny stays what it was.
    #[test]
    fn escalation_never_relaxes_a_decision() {
        let g = guarded_gate("default");
        let ask = g.decide(
            "guarded",
            Some(ToolClass::Process),
            false,
            CallRisk::Normal,
            "s",
        );
        assert_eq!(g.escalate("guarded", "action", ask.clone(), "x"), ask);
        let deny = Decision::Deny { reason: "r".into() };
        assert_eq!(g.escalate("guarded", "action", deny.clone(), "x"), deny);
        assert_eq!(
            g.escalate("unregistered", "action", Decision::Allow, ""),
            Decision::Allow
        );
    }

    /// Changing mode keeps the session grant.
    #[test]
    fn set_mode_keeps_the_session_grant() {
        let mut g = guarded_gate("default");
        g.allow_tool_for_session("guarded");
        g.set_mode(PermissionMode::from_str("auto"));
        assert_eq!(g.mode(), PermissionMode::Auto);
        assert_eq!(
            g.decide(
                "guarded",
                Some(ToolClass::Process),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Allow
        );
    }
    #[test]
    fn revoking_a_tool_also_revokes_the_policys_related_grants() {
        let mut g = guarded_gate("auto");
        g.allow_tool_for_session("guarded");
        g.allow_tool_for_session("guarded:extra");
        g.allow_tool_for_session("bash");
        assert!(g.revoke_grants("guarded"));
        // Back to asking, and unrelated grants are left alone.
        assert!(matches!(
            g.decide(
                "guarded",
                Some(ToolClass::Process),
                false,
                CallRisk::Normal,
                ""
            ),
            Decision::Ask { .. }
        ));
        assert!(!g.rules.allow.contains("guarded:extra"));
        assert!(g.decide("bash", None, false, CallRisk::Normal, "") == Decision::Allow);
        // Nothing to revoke the second time → no spurious notice.
        assert!(!g.revoke_grants("guarded"));
    }
}
