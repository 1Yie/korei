#!/usr/bin/env bash
# Kernel closure law — the standalone-repo version of check-deps.sh. Needs no
# product table: it asserts the closure itself. The kernel crates may depend on
# each other and on nothing else inside this workspace, and their normal
# dependency tree must stay free of the heavy provider stack.
#
# When the kernel moves to its own repository this file moves with it; the
# product repository keeps scripts/check-deps.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

declare -A ALLOW=(
  [agent-ipc]=""
  [agent-llm-core]=""
  [agent-tool]="agent-llm-core"
  [agent-kernel]="agent-ipc agent-llm-core agent-tool"
)
# Third-party crates the kernel closure must never pull in.
FORBIDDEN="rig-core reqwest keyring toml toml_edit"

fail=0
meta=$(cargo metadata --no-deps --format-version 1)
for name in "${!ALLOW[@]}"; do
  deps=$(jq -r --arg n "$name" '
    ([.packages[].name] | unique) as $ws
    | .packages[] | select(.name == $n)
    | [ .dependencies[]
        | select(.kind != "dev" and (.name as $d | $ws | index($d)))
        | .name ] | unique | join(" ")' <<<"$meta")
  for dep in $deps; do
    if [[ " ${ALLOW[$name]} " != *" $dep "* ]]; then
      echo "FAIL  $name -> $dep is outside the kernel closure (allowed: ${ALLOW[$name]:-none})"
      fail=1
    fi
  done
  tree=$(cargo tree -p "$name" -e normal --prefix none 2>/dev/null | awk '{print $1}' | sort -u)
  for bad in $FORBIDDEN; do
    if grep -qx "$bad" <<<"$tree"; then
      echo "FAIL  $name pulls in $bad"
      fail=1
    fi
  done
done

[[ $fail -eq 0 ]] && echo "kernel closure law: ok"
exit $fail
