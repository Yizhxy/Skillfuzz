#!/usr/bin/env bash
# Smoke-test the LLM endpoints used by fuzzer.sh.
#
# Mirrors the env-var defaults from ../fuzzer.sh so this script can be run
# standalone:
#     bash test/test_llms.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------
# Anthropic / Claude Code (used by Harbor's claude-code agent)
# --------------------------------------------------------------------
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.v3.cm}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-X1mHup3ki3dfezMH502a167c3fB54d1284Ea28EfF9B5E3Ca}"
export HARBOR_MODEL="${HARBOR_MODEL:-claude-haiku-4-5-20251001}"

# --------------------------------------------------------------------
# Eval / Mutate LLM (OpenAI-compatible endpoint, used by fuzzer.py)
# --------------------------------------------------------------------
export LLM_API_BASE="${LLM_API_BASE:-https://api.v3.cm}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-X1mHup3ki3dfezMH502a167c3fB54d1284Ea28EfF9B5E3Ca}"
export LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY}}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"

exec python3 "${SCRIPT_DIR}/test_llms.py"
