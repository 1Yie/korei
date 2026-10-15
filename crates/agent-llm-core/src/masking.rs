//! `masking.rs` — egress secret masker.
//!
//! Every outbound request body and every tool result passes through the replacer:
//! secrets leak most often via `env`/`.env` reads landing in a prompt. Sandbox env
//! sanitization is the first wall, this is the second — a match becomes
//! `[REDACTED_<kind>]`.

/// Resolved secret values + the shapes that catch them.
pub struct EgressMasker {
    /// Concrete secret strings resolved from `keyring:`/`env:` config —
    /// the exact values to scrub (never the `keyring:` handle itself).
    secrets: Vec<String>,
}

/// Patterns marking text as secret-shaped — matched case-insensitively on
/// the *content*, not just names (a value containing `sk-ant-…` is a leak
/// even if the field was named `note`).
const SECRET_SHAPES: &[&str] = &[
    "sk-",
    "sk-ant-",
    "sk-proj-",
    "ghp_",
    "gho_",
    "github_pat_",
    "xai-",
    "glpat-",
    "AKIA",
    "AIza",
    "ya29.",
    "dop_v1_",
    "-----BEGIN",
    "PRIVATE KEY-----",
];

/// Whole field names whose values are always scrubbed (`"api_key": …`).
const SECRET_FIELDS: &[&str] = &[
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "pwd",
    "api_key",
    "apikey",
    "api_secret",
    "private_key",
    "client_secret",
    "access_key",
    "secret_key",
    "access_token",
    "refresh_token",
    "auth_token",
    "id_token",
    "session_key",
    "encryption_key",
];

/// Suffixes that mark a longer field name as secret too — `AWS_SECRET_ACCESS_KEY`,
/// `my_api_token`, `db_password`. Matched on the **last identifier** before the
/// `:`/`=`, never as a substring of the whole line, so `VIRTUAL_KEY`/`key_input`
/// don't trip it (they aren't the field-name position; see `mask_field_values`).
const SECRET_SUFFIXES: &[&str] = &[
    "_key",
    "_token",
    "_secret",
    "_password",
    "_passwd",
    "_apikey",
];

/// Rust/other keywords that start a statement — if the left side of `:`/`=`
/// begins with one, it is a declaration/signature, not a `field: value` secret.
/// `fn key_input(vk: u16`, `let flag = X`, `use …::KEY` all bail here, which is
/// what stopped the `_key`-in-`VIRTUAL_KEY` self-mask that wrote
/// `[REDACTED_SECRET]` into `windows.rs` history.
const STMT_PREFIXES: &[&str] = &[
    "fn ", "let ", "const ", "static ", "pub ", "use ", "type ", "struct ", "enum ", "impl ",
    "mod ", "match ", "if ", "for ", "while ", "return ", "async ", "unsafe ", "extern ", "trait ",
    "where ",
];

impl EgressMasker {
    /// Build from the resolved secret strings (post `env:`/`keyring:`).
    pub fn new(secrets: Vec<String>) -> Self {
        // Keep only non-trivial secrets — a 4-char value would over-mask.
        let secrets = secrets.into_iter().filter(|s| s.len() >= 8).collect();
        Self { secrets }
    }

    /// Scrub `text` — exact secrets → `[REDACTED_SECRET]`, shape matches →
    /// `[REDACTED_<kind>]`. Returns the sanitized string + whether it changed.
    pub fn scrub(&self, text: &str) -> (String, bool) {
        let mut out = text.to_string();
        let mut changed = false;
        // 1. Exact resolved secrets — highest confidence.
        for s in &self.secrets {
            if out.contains(s.as_str()) {
                out = out.replace(s.as_str(), "[REDACTED_SECRET]");
                changed = true;
            }
        }
        // 2. Shape matches — the value looks like a key/token.
        for shape in SECRET_SHAPES {
            if let Some(masked) = mask_shape(&out, shape) {
                out = masked;
                changed = true;
            }
        }
        // 3. Field-name secrets — `"api_key": "..."` → scrub the value.
        out = mask_field_values(&out);
        (out, changed)
    }
}

/// Replace `shape`-prefixed tokens with `[REDACTED_<shape>]` — any `sk-…` run becomes
/// `[REDACTED_API_KEY]`.
///
/// A shape must start at a word boundary, else `task-force`/`disk-`/`risk-` get
/// mangled by the `sk-` match. `-----BEGIN` opens a PEM block and the whole block is
/// redacted through `-----END …-----` — masking the header alone leaks the base64
/// material.
fn mask_shape(text: &str, shape: &str) -> Option<String> {
    let needle = shape.to_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut any = false;
    loop {
        // Offsets are measured on `rest` itself — never on a lowercased copy.
        // `to_lowercase()` changes byte lengths (`ẞ`→`ß` shrinks, `İ`→`i̇`
        // grows), so an index found in the folded string is NOT a valid index
        // into the original: slicing `rest[..idx]` with it lands mid-char
        // (panic) or silently checks the wrong boundary char.
        let Some(idx) = find_ascii_ci(rest, &needle) else {
            break;
        };

        // Word-boundary check: the char before the match must be a boundary
        // (start, whitespace, quote, or non-alphanumeric punctuation), else
        // this `sk-` is inside a longer word like `task-`/`disk-`/`msk-`.
        let boundary_ok = idx == 0
            || rest[..idx]
                .chars()
                .last()
                .map(|c| !c.is_alphanumeric() && c != '-' && c != '_')
                .unwrap_or(true);
        if !boundary_ok {
            // Not a token start — emit through this char and keep scanning.
            let adv = idx + needle.len();
            out.push_str(&rest[..adv]);
            rest = &rest[adv..];
            continue;
        }

        out.push_str(&rest[..idx]);
        let tail = &rest[idx..];

        // PEM block: redact through the matching `-----END …-----` line so
        // the key material can't leak between BEGIN and END.
        if needle.starts_with("-----begin") {
            if let Some(end_idx) = find_ascii_ci(tail, "-----end") {
                let after_end = &tail[end_idx..];
                let line_end = after_end.find('\n').unwrap_or(after_end.len());
                out.push_str("[REDACTED_PEM_BLOCK]\n");
                rest = &after_end[line_end..];
                any = true;
                continue;
            }
            // Unterminated block — redact to end of text.
            out.push_str("[REDACTED_PEM_BLOCK]");
            rest = "";
            any = true;
            break;
        }

        // Ordinary token: consume shape + the following non-space run.
        let end = tail
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
            .unwrap_or(tail.len());
        out.push_str("[REDACTED_API_KEY]");
        rest = &tail[end..];
        any = true;
    }
    out.push_str(rest);
    any.then_some(out)
}

/// First offset of `needle` in `haystack`, ASCII-case-insensitive — every
/// shape in `SECRET_SHAPES` is ASCII, so `eq_ignore_ascii_case` is the whole
/// semantics (non-ASCII bytes can never equal the needle's). Returns a real
/// char boundary of `haystack` because the scan only starts at boundaries.
fn find_ascii_ci(haystack: &str, needle: &str) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    haystack.char_indices().map(|(i, _)| i).find(|&i| {
        haystack[i..]
            .get(..needle.len())
            .is_some_and(|w| w.eq_ignore_ascii_case(needle))
    })
}

/// `"<field>": "<value>"` / `<field>=<value>` → scrub value when field is
/// secret-shaped.
fn mask_field_values(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        // Only inspect `field: value` / `field = value` lines — extract the
        // field name as the LAST identifier before the `:`/`=` and compare it
        // as a whole token. Matching `contains` over the whole line mangles
        // code (`_key` ⊂ `VIRTUAL_KEY`/`key_input`), and the resulting
        // `[REDACTED_SECRET]` leaks back into model history as literal text.
        let sep = line
            .rfind(|c: char| c == 61u8 as char)
            .or_else(|| line.find(|c: char| c == 58u8 as char));
        if let Some(pos) = sep {
            let lhs = &line[..pos];
            // Skip statement/decl context: `fn key_input(vk: ...`,
            // `let flag = ...`, `use …::KEY`, `#define X ...` are code, not
            // `field = secret` config lines. A leading statement keyword or a
            // `#` directive marks code. NOTE: `{`/`(` must NOT count as code —
            // JSON payloads (`{"cmd":"export KEY=…"}`) and call args start with
            // them, and they are exactly the exfiltration path being scrubbed.
            let lead = lhs.trim_start();
            let is_stmt =
                STMT_PREFIXES.iter().any(|p| lead.starts_with(p)) || lead.starts_with('#');
            let field = last_identifier(lhs);
            let is_secret_field = field
                .map(|f| {
                    let fl = f.to_lowercase();
                    SECRET_FIELDS.contains(&fl.as_str())
                        || SECRET_SUFFIXES
                            .iter()
                            .any(|s| fl.len() > s.len() && fl.ends_with(s))
                })
                .unwrap_or(false)
                && !is_stmt;
            if is_secret_field {
                let (head, val) = line.split_at(pos + 1);
                let v = val.trim();
                if !v.is_empty() && !v.starts_with('[') && v.len() >= 6 {
                    out.push_str(head);
                    out.push_str(" [REDACTED_SECRET]\n");
                    continue;
                }
            }
        }
        out.push_str(line);
    }
    out
}

/// The last `[A-Za-z0-9_]` run in `lhs` (the text before the `:`/`=`) — the
/// field name. `"access_key" ` → `access_key`; `fn key_input(vk` → `vk` (so
/// `vk: [REDACTED_SECRET]
/// "let flag "` → `flag`).
fn last_identifier(lhs: &str) -> Option<&str> {
    let trimmed = lhs.trim_end_matches(|c: char| !(c.is_ascii_alphanumeric() || c == 95u8 as char));
    // `rfind` gives a char's byte START — `i + 1` slices inside any
    // multi-byte char (CJK before a `:`/`=` crashed the process under
    // panic = "abort"). Skip the whole char instead.
    let start = trimmed
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_'))
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let ident = &trimmed[start..];
    (!ident.is_empty() && ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .then_some(ident)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_secret_scrubbed() {
        let m = EgressMasker::new(vec!["xai-real-key-123".into()]);
        let (out, changed) = m.scrub("the key is xai-real-key-123 ok");
        assert!(changed);
        assert!(out.contains("[REDACTED_SECRET]"));
        assert!(!out.contains("xai-real-key-123"));
    }

    /// `to_lowercase()` changes byte lengths (`ẞ`→`ß` shrinks one byte), so a
    /// match offset taken in the lowercased copy is not a valid index into
    /// the original string. Before the fix this either panicked (mid-char
    /// slice, under `panic = "abort"` = process death) or skipped the shape —
    /// here the drift makes the boundary check read `r` and jump past `sk-`.
    #[test]
    fn non_ascii_text_before_a_shape_neither_panics_nor_leaks() {
        let m = EgressMasker::new(vec![]);
        let (out, changed) = m.scrub("ẞẞ before sk-ant-secretkey999 after");
        assert!(changed);
        assert!(
            !out.contains("sk-ant"),
            "secret leaked through an offset-drifted boundary check: {out}"
        );
        // And a shape that sits inside a word must still NOT be masked.
        let (out2, changed2) = m.scrub("the word task-force stays");
        assert_eq!(out2, "the word task-force stays");
        assert!(!changed2);
    }

    #[test]
    fn shape_scrubbed() {
        let m = EgressMasker::new(vec![]);
        let (out, _) = m.scrub("call with sk-ant-abc123 done");
        assert!(out.contains("[REDACTED_API_KEY]"));
    }

    #[test]
    fn field_value_scrubbed() {
        let m = EgressMasker::new(vec![]);
        let (out, _) = m.scrub("api_key: very_secret_value_here");
        assert!(out.contains("[REDACTED_SECRET]"));
    }

    /// The two coredumps: `field: value` where the LHS ends in a CJK char
    /// right before the identifier (`让`/`）`). `rfind` returned the char's
    /// byte start and `i + 1` sliced inside it → panic → process abort.
    #[test]
    fn cjk_before_a_field_name_does_not_panic() {
        let out = mask_field_values("中文api_key: very_secret_value_here\n");
        assert!(out.contains("[REDACTED_SECRET]"), "{out}");
        // Pure-CJK lhs has no identifier — passes through untouched.
        let out = mask_field_values("备注说明: some_value\n");
        assert_eq!(out, "备注说明: some_value\n");
    }

    #[test]
    fn clean_text_untouched() {
        let m = EgressMasker::new(vec![]);
        let (out, changed) = m.scrub("fn main() { println!(\"hi\"); }");
        assert!(!changed);
        assert_eq!(out, "fn main() { println!(\"hi\"); }");
    }

    #[test]
    fn word_boundary_spares_task_force() {
        // `sk-` inside `task-force`, `disk-`, `risk-` must NOT be masked —
        // a substring match with no token boundary mangles normal prose.
        let m = EgressMasker::new(vec![]);
        for w in ["task-force", "disk-full", "risk-reward", "msk-edge"] {
            let (out, _) = m.scrub(w);
            assert_eq!(out, w, "`{w}` was mangled by the sk- shape");
        }
    }

    #[test]
    fn pem_block_fully_redacted() {
        // `-----BEGIN` opens a multi-line key block — masking only the
        // header line would leak the base64 material between BEGIN and END.
        let m = EgressMasker::new(vec![]);
        let pem = "here\n-----BEGIN PRIVATE KEY-----\nMIIabc123==\nMIIdef456==\n-----END PRIVATE KEY-----\nafter";
        let (out, _) = m.scrub(pem);
        assert!(!out.contains("MIIabc123"), "key material leaked");
        assert!(!out.contains("MIIdef456"), "key material leaked");
        assert!(out.contains("[REDACTED_PEM_BLOCK]"));
        assert!(out.contains("after"));
    }

    #[test]
    fn tool_call_arguments_are_scrubbed() {
        // A tool that read `.env` puts the secret into the NEXT request's
        // tool_calls[].arguments — this is the exfiltration path the masker
        // originally missed. Verify the field is covered end-to-end.
        let m = EgressMasker::new(vec![]);
        let (out, _) = m.scrub("{\"cmd\":\"export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI\"}");
        assert!(!out.contains("wJalrXUtnFEMI"));
    }

    /// Regression for the `windows.rs` corruption: `_key` inside `VIRTUAL_KEY`,
    /// `INPUT_KEYBOARD`, `key_input`, `wVk` used to trip `contains("_key")` and
    /// rewrite whole declarations into `[REDACTED_SECRET]`, which the model then
    /// echoed back as literal source. Statement/decl context must NOT be masked.
    #[test]
    fn code_declarations_are_not_masked() {
        let m = EgressMasker::new(vec![]);
        for src in [
            "fn key_input(vk: VIRTUAL_KEY, flags: KEYBD_EVENT_FLAGS, unicode: u16) -> INPUT {",
            "    let flag = KEYBD_EVENT_FLAGS(0);",
            "        r#type: INPUT_KEYBOARD,",
            "use windows::Win32::UI::Input::KeyboardAndMouse::VIRTUAL_KEY;",
            "    pub vk: VIRTUAL_KEY,",
            "struct S { access_key: u32 }",
        ] {
            let (out, _) = m.scrub(src);
            assert_eq!(out, src, "code line was mangled: {src:?}");
        }
    }

    /// The `_key`/`_token`/`_secret` suffix still catches long env-var-style
    /// names on the LAST identifier before `=`/`:` — `AWS_SECRET_ACCESS_KEY`,
    /// `MY_DB_PASSWORD`, json `"refresh_token": "…"`.
    #[test]
    fn secret_suffix_fields_still_masked() {
        let m = EgressMasker::new(vec![]);
        let (out, _) = m.scrub("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI123");
        assert!(!out.contains("wJalrXUtnFEMI"), "suffix field leaked: {out}");
        let (out2, _) = m.scrub("{\"refresh_token\": \"tok_abc123456\"}");
        assert!(!out2.contains("tok_abc123456"), "json field leaked: {out2}");
        // exact whole-name `password` too
        let (out3, _) = m.scrub("password = hunter2_secret");
        assert!(!out3.contains("hunter2_secret"));
    }
}
