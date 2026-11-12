//! Offline tool schemas with acyclic JSON Pointer references.
//!
//! Dynamic/recursive references, anchors, resource IDs, regex constraints,
//! custom vocabularies and content schemas are unsupported. Annotations and
//! literal values are data; only schema-bearing keyword positions are followed.
//! `format` is an annotation; assertion vocabularies are unsupported.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;

use serde_json::Value;

pub const MAX_ARGUMENT_BYTES: usize = 64 * 1024;
pub const MAX_SCHEMA_BYTES: usize = 256 * 1024;
pub const MAX_NODES: usize = 16_384;
pub const MAX_DEPTH: usize = 64;
pub const MAX_EXPANDED_NODES: usize = 4_096;
pub const MAX_EXPANDED_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_VALIDATION_WORK: usize = 16_000_000;

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("invalid JSON: {0}")]
    Json(String),
    #[error("invalid tool schema: {0}")]
    Schema(String),
    #[error("invalid tool arguments: {0}")]
    Arguments(String),
    #[error("unsupported tool schema: {0}")]
    Unsupported(String),
    #[error("validation limit exceeded: {0}")]
    Limit(&'static str),
}

impl From<ValidationError> for String {
    fn from(error: ValidationError) -> Self {
        error.to_string()
    }
}

struct ByteBudget(usize);
impl Write for ByteBudget {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.0 {
            return Err(std::io::Error::other("JSON byte limit"));
        }
        self.0 -= bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn bounded_value(value: &Value, max_bytes: usize) -> Result<usize, ValidationError> {
    let mut pending = vec![(value, 0)];
    let mut nodes = 0;
    while let Some((value, depth)) = pending.pop() {
        nodes += 1;
        if nodes > MAX_NODES || depth > MAX_DEPTH {
            return Err(ValidationError::Limit("JSON nodes/depth"));
        }
        match value {
            Value::Object(object) => pending.extend(object.values().map(|v| (v, depth + 1))),
            Value::Array(array) => pending.extend(array.iter().map(|v| (v, depth + 1))),
            _ => {}
        }
        if pending.len() > MAX_NODES {
            return Err(ValidationError::Limit("JSON width"));
        }
    }
    serde_json::to_writer(ByteBudget(max_bytes), value)
        .map_err(|_| ValidationError::Limit("JSON bytes"))?;
    Ok(nodes)
}

pub fn parse_arguments(arguments: &str) -> Result<Value, ValidationError> {
    if arguments.len() > MAX_ARGUMENT_BYTES {
        return Err(ValidationError::Limit("argument bytes"));
    }
    let value =
        serde_json::from_str(arguments).map_err(|e| ValidationError::Json(e.to_string()))?;
    bounded_value(&value, MAX_ARGUMENT_BYTES)?;
    Ok(value)
}

#[derive(Default)]
struct Node {
    edges: Vec<String>,
}

fn pointer(parent: &str, key: &str) -> String {
    format!("{parent}/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn collect(
    schema: &Value,
    path: String,
    nodes: &mut BTreeMap<String, Node>,
    unique: &mut bool,
) -> Result<(), ValidationError> {
    let mut edges = Vec::new();
    let Some(object) = schema.as_object() else {
        if schema.is_boolean() {
            nodes.insert(path, Node::default());
            return Ok(());
        }
        return Err(ValidationError::Schema(format!(
            "expected schema at #{path}"
        )));
    };
    for keyword in [
        "$id",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$recursiveRef",
        "$recursiveAnchor",
        "$vocabulary",
        "pattern",
        "contentSchema",
    ] {
        if object.contains_key(keyword) {
            return Err(ValidationError::Unsupported(keyword.into()));
        }
    }
    if let Some(patterns) = object.get("patternProperties") {
        if patterns.as_object().is_none_or(|map| !map.is_empty()) {
            return Err(ValidationError::Unsupported("patternProperties".into()));
        }
    }
    if let Some(dialect) = object.get("$schema") {
        if !matches!(
            dialect.as_str(),
            Some(
                "https://json-schema.org/draft/2020-12/schema"
                    | "https://json-schema.org/draft/2019-09/schema"
                    | "http://json-schema.org/draft-07/schema#"
                    | "http://json-schema.org/draft-06/schema#"
                    | "http://json-schema.org/draft-04/schema#"
            )
        ) {
            return Err(ValidationError::Unsupported(
                "unknown schema dialect".into(),
            ));
        }
    }
    if let Some(reference) = object.get("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| ValidationError::Schema("$ref must be a string".into()))?;
        if reference != "#" && (!reference.starts_with("#/") || reference.contains('%')) {
            return Err(ValidationError::Unsupported(
                "only local JSON Pointer references are supported".into(),
            ));
        }
        edges.push(reference[1..].to_string());
    }
    *unique |= object.get("uniqueItems") == Some(&Value::Bool(true));
    for (keyword, child) in object {
        let base = pointer(&path, keyword);
        match keyword.as_str() {
            "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas"
            | "dependencies" => {
                let map = child.as_object().ok_or_else(|| {
                    ValidationError::Schema(format!("{keyword} must be an object"))
                })?;
                for (name, child) in map {
                    if keyword == "dependencies" && child.is_array() {
                        continue;
                    }
                    let child_path = pointer(&base, name);
                    collect(child, child_path.clone(), nodes, unique)?;
                    edges.push(child_path);
                }
            }
            "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                let children = child.as_array().ok_or_else(|| {
                    ValidationError::Schema(format!("{keyword} must be an array"))
                })?;
                for (index, child) in children.iter().enumerate() {
                    let child_path = pointer(&base, &index.to_string());
                    collect(child, child_path.clone(), nodes, unique)?;
                    edges.push(child_path);
                }
            }
            "items" if child.is_array() => {
                for (index, child) in child.as_array().unwrap().iter().enumerate() {
                    let child_path = pointer(&base, &index.to_string());
                    collect(child, child_path.clone(), nodes, unique)?;
                    edges.push(child_path);
                }
            }
            "items"
            | "additionalItems"
            | "additionalProperties"
            | "unevaluatedItems"
            | "unevaluatedProperties"
            | "propertyNames"
            | "contains"
            | "not"
            | "if"
            | "then"
            | "else" => {
                collect(child, base.clone(), nodes, unique)?;
                edges.push(base);
            }
            _ => {}
        }
    }
    nodes.insert(path, Node { edges });
    Ok(())
}

fn expansion(
    key: &str,
    nodes: &BTreeMap<String, Node>,
    visiting: &mut Vec<String>,
    memo: &mut HashMap<String, (usize, usize)>,
) -> Result<(usize, usize), ValidationError> {
    if visiting.iter().any(|p| p == key) {
        return Err(ValidationError::Unsupported(
            "cyclic local schema references".into(),
        ));
    }
    if visiting.len() >= MAX_DEPTH {
        return Err(ValidationError::Limit("expanded schema depth"));
    }
    if let Some(&(cost, height)) = memo.get(key) {
        if visiting.len() + height > MAX_DEPTH {
            return Err(ValidationError::Limit("expanded schema depth"));
        }
        return Ok((cost, height));
    }
    let node = nodes.get(key).ok_or_else(|| {
        ValidationError::Unsupported(format!("reference target #{key} is not a schema position"))
    })?;
    visiting.push(key.to_string());
    let mut cost = 1usize;
    let mut height = 1usize;
    for edge in &node.edges {
        let (child_cost, child_height) = expansion(edge, nodes, visiting, memo)?;
        height = height.max(child_height + 1);
        if height > MAX_DEPTH {
            return Err(ValidationError::Limit("expanded schema depth"));
        }
        cost = cost
            .checked_add(child_cost)
            .filter(|cost| *cost <= MAX_EXPANDED_NODES)
            .ok_or(ValidationError::Limit("expanded schema nodes"))?;
    }
    visiting.pop();
    memo.insert(key.to_string(), (cost, height));
    Ok((cost, height))
}

pub struct BoundedValidator {
    validator: jsonschema::Validator,
    work: usize,
    unique: bool,
}

pub fn validator(schema: &Value) -> Result<BoundedValidator, ValidationError> {
    let raw_nodes = bounded_value(schema, MAX_SCHEMA_BYTES)?;
    let mut nodes = BTreeMap::new();
    let mut unique = false;
    collect(schema, String::new(), &mut nodes, &mut unique)?;
    let mut memo = HashMap::new();
    let mut expanded = 0;
    for key in nodes.keys() {
        expanded = expanded.max(expansion(key, &nodes, &mut Vec::new(), &mut memo)?.0);
    }
    let mut bytes = ByteBudget(MAX_SCHEMA_BYTES);
    serde_json::to_writer(&mut bytes, schema)
        .map_err(|_| ValidationError::Limit("schema bytes"))?;
    if (MAX_SCHEMA_BYTES - bytes.0)
        .checked_mul(expanded)
        .is_none_or(|size| size > MAX_EXPANDED_BYTES)
    {
        return Err(ValidationError::Limit("expanded schema bytes"));
    }
    let work = expanded
        .checked_mul(raw_nodes)
        .ok_or(ValidationError::Limit("schema work"))?;
    if work > MAX_VALIDATION_WORK {
        return Err(ValidationError::Limit("schema work"));
    }
    let validator = jsonschema::options()
        .should_validate_formats(false)
        .build(schema)
        .map_err(|e| ValidationError::Schema(e.to_string()))?;
    Ok(BoundedValidator {
        validator,
        work,
        unique,
    })
}

impl BoundedValidator {
    pub fn validate(&self, arguments: &Value) -> Result<(), ValidationError> {
        let nodes = bounded_value(arguments, MAX_ARGUMENT_BYTES)?;
        let work = self
            .work
            .checked_mul(nodes)
            .and_then(|n| {
                if self.unique {
                    n.checked_mul(nodes)
                } else {
                    Some(n)
                }
            })
            .ok_or(ValidationError::Limit("validation work"))?;
        if work > MAX_VALIDATION_WORK {
            return Err(ValidationError::Limit("validation work"));
        }
        self.validator
            .validate(arguments)
            .map_err(|e| ValidationError::Arguments(e.to_string()))
    }
}

pub fn validate(schema: &Value, arguments: &Value) -> Result<(), ValidationError> {
    validator(schema)?.validate(arguments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cycles_and_dynamic_references_are_rejected_before_library_compilation() {
        for schema in [
            json!({"allOf":[{"$ref":"#"}]}),
            json!({"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}}}),
            json!({"$dynamicRef":"#"}),
            json!({"$recursiveRef":"#"}),
            json!({"$ref":"file:///etc/passwd"}),
            json!({"$ref":"https://example.invalid/s"}),
        ] {
            assert!(matches!(
                validator(&schema),
                Err(ValidationError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn repeated_dag_expansion_is_bounded_and_ordinary_local_references_work() {
        let mut defs = serde_json::Map::new();
        defs.insert("n0".into(), json!({"type":"integer"}));
        for i in 1..16 {
            let reference = format!("#/$defs/n{}", i - 1);
            defs.insert(
                format!("n{i}"),
                json!({"allOf":[{"$ref":reference},{"$ref":reference}]}),
            );
        }
        assert!(matches!(
            validator(&json!({"$defs":defs,"$ref":"#/$defs/n15"})),
            Err(ValidationError::Limit(_))
        ));
        let schema = json!({"$defs":{"integer":{"type":"integer"}},"$ref":"#/$defs/integer"});
        assert!(validate(&schema, &json!(1)).is_ok());
        assert!(matches!(
            validate(&schema, &json!("no")),
            Err(ValidationError::Arguments(_))
        ));
    }

    #[test]
    fn literal_reference_fields_are_data_and_non_schema_targets_are_rejected() {
        let literal = json!({"$ref":"file:///literal","$dynamicRef":"#","$id":"literal"});
        let schema = json!({
            "type":"object","properties":{"$ref":{"type":"string"}},
            "default":literal,"examples":[literal],"const":literal
        });
        assert!(validate(&schema, &literal).is_ok());
        assert!(matches!(
            validator(&json!({"default":{"$ref":"#"},"$ref":"#/default"})),
            Err(ValidationError::Unsupported(_))
        ));
    }

    #[test]
    fn argument_resources_and_unbounded_regex_features_are_rejected() {
        assert!(parse_arguments(&" ".repeat(MAX_ARGUMENT_BYTES + 1)).is_err());
        assert!(matches!(
            validate(&json!({}), &json!(vec![0; MAX_NODES + 1])),
            Err(ValidationError::Limit(_))
        ));
        let mut deep = json!(0);
        for _ in 0..MAX_DEPTH + 1 {
            deep = json!([deep]);
        }
        assert!(matches!(
            validate(&json!({}), &deep),
            Err(ValidationError::Limit(_))
        ));
        assert!(matches!(
            validator(&json!({"pattern":"(a+)+$"})),
            Err(ValidationError::Unsupported(_))
        ));
    }
}
