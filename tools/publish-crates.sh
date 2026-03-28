#!/usr/bin/env bash

set -euo pipefail

readonly crates=(
  vision-calibration-core
  vision-geometry
  vision-calibration-linear
  vision-calibration-optim
  vision-calibration-pipeline
  vision-mvg
  vision-calibration
)

usage() {
  cat <<'EOF'
Usage: tools/publish-crates.sh <list|check|package|publish> [extra cargo args...]

Commands:
  list      Print crates in publish order.
  check     Validate publishable crate manifests with cargo package --list.
  package   Run cargo package for each publishable Rust crate.
  publish   Run cargo publish for each publishable Rust crate.

Examples:
  tools/publish-crates.sh list
  tools/publish-crates.sh check
  tools/publish-crates.sh package --no-verify
  tools/publish-crates.sh publish
EOF
}

run_manifest_checks() {
  for crate in "${crates[@]}"; do
    echo "Checking ${crate}..."
    cargo package -p "${crate}" --locked --no-verify --list "$@" >/dev/null
  done
}

run_cargo_subcommand() {
  local subcommand="$1"
  shift

  for i in "${!crates[@]}"; do
    local crate="${crates[$i]}"
    if [[ "${subcommand}" == "package" ]]; then
      echo "Packaging ${crate}..."
    else
      echo "Publishing ${crate}..."
    fi
    cargo "${subcommand}" -p "${crate}" --locked "$@"

    if [[ "${subcommand}" == "publish" ]] && (( i + 1 < ${#crates[@]} )); then
      # Give the crates.io index time to propagate before dependents publish.
      sleep 20
    fi
  done
}

main() {
  local command="${1:-list}"
  if [[ "$#" -gt 0 ]]; then
    shift
  fi

  case "${command}" in
    list)
      printf '%s\n' "${crates[@]}"
      ;;
    check)
      run_manifest_checks "$@"
      ;;
    package)
      run_cargo_subcommand package "$@"
      ;;
    publish)
      run_cargo_subcommand publish "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
