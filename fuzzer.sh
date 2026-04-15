#!/usr/bin/env bash
# Launch the skill-workflow fuzzer.
#
# Usage:
#   bash fuzzer.sh                       # fuzz all tasks in tasks/
#   FUZZ_TASKS=citation-check bash fuzzer.sh
#   FUZZ_MAX_ITERS=3 bash fuzzer.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# --------------------------------------------------------------------
# Enable BuildKit (required for Dockerfile heredoc syntax)
# --------------------------------------------------------------------
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# --------------------------------------------------------------------
# TLS trust store
# Conda Pythons ship their own OpenSSL and don't read the system CA bundle,
# so urllib in fuzzer.py would fail with CERTIFICATE_VERIFY_FAILED. Point it
# at the system bundle (override by exporting SSL_CERT_FILE before running).
# --------------------------------------------------------------------
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"

# --------------------------------------------------------------------
# Anthropic / Claude Code (used by Harbor's claude-code agent)
# --------------------------------------------------------------------
export ANTHROPIC_BASE_URL="https://api.v3.cm"
export ANTHROPIC_API_KEY="sk-X1mHup3ki3dfezMH502a167c3fB54d1284Ea28EfF9B5E3Ca"

# --------------------------------------------------------------------
# Eval / Mutate LLM (OpenAI-compatible endpoint, used by fuzzer.py)
# --------------------------------------------------------------------
export LLM_API_BASE="https://api.v3.cm"
export OPENAI_API_KEY="sk-X1mHup3ki3dfezMH502a167c3fB54d1284Ea28EfF9B5E3Ca"
export LLM_API_KEY="${OPENAI_API_KEY}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
export LLM_TIMEOUT_SEC="${LLM_TIMEOUT_SEC:-120}"

# Set FUZZ_JSON_MODE=0 if the gateway does not support response_format=json_object.
export FUZZ_JSON_MODE="${FUZZ_JSON_MODE:-1}"
export FUZZ_EVAL_TEMP="${FUZZ_EVAL_TEMP:-0.2}"
export FUZZ_MUTATE_TEMP="${FUZZ_MUTATE_TEMP:-0.8}"

# --------------------------------------------------------------------
# Harbor runner
# --------------------------------------------------------------------
export HARBOR_BIN="${HARBOR_BIN:-harbor}"
export HARBOR_CWD="${HARBOR_CWD:-/data/hxy/skillsbench-main}"
export HARBOR_AGENT="${HARBOR_AGENT:-claude-code}"
export HARBOR_MODEL="${HARBOR_MODEL:-claude-haiku-4-5-20251001}"
export HARBOR_DELETE="${HARBOR_DELETE:-1}"

# --------------------------------------------------------------------
# Fuzz loop
# --------------------------------------------------------------------
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export FUZZ_WORK_ROOT="${FUZZ_WORK_ROOT:-${SCRIPT_DIR}/fuzz_runs_${RUN_TS}}"
export FUZZ_MAX_ITERS="${FUZZ_MAX_ITERS:-5}"
# How many tasks to fuzz in parallel (1 = sequential, default).
export FUZZ_PARALLEL="${FUZZ_PARALLEL:-1}"
# Optional: comma-separated task names to restrict which tasks are fuzzed.
# export FUZZ_TASKS="citation-check,hvac-control"

# Artifact truncation caps (chars) sent to the eval LLM.
export FUZZ_MAX_TRAJ_CHARS="${FUZZ_MAX_TRAJ_CHARS:-60000}"
export FUZZ_MAX_VERIFIER_CHARS="${FUZZ_MAX_VERIFIER_CHARS:-4000}"
export FUZZ_MAX_HARBOR_CHARS="${FUZZ_MAX_HARBOR_CHARS:-4000}"

mkdir -p "${FUZZ_WORK_ROOT}"

# --------------------------------------------------------------------
# Logging — all output goes to logs/<timestamp>.log
# Set FUZZ_FOREGROUND=1 to also stream to terminal (default: background).
# --------------------------------------------------------------------
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_TS}.log"
PID_FILE="${LOG_DIR}/${RUN_TS}.pid"

echo "[fuzzer.sh] work_root   = ${FUZZ_WORK_ROOT}"
echo "[fuzzer.sh] max_iters   = ${FUZZ_MAX_ITERS}"
echo "[fuzzer.sh] parallel    = ${FUZZ_PARALLEL}"
echo "[fuzzer.sh] llm_model   = ${LLM_MODEL}"
echo "[fuzzer.sh] harbor      = ${HARBOR_BIN} (agent=${HARBOR_AGENT}, model=${HARBOR_MODEL})"
echo "[fuzzer.sh] log_file    = ${LOG_FILE}"
if [[ -n "${FUZZ_TASKS:-}" ]]; then
    echo "[fuzzer.sh] task filter = ${FUZZ_TASKS}"
fi

if [[ "${FUZZ_FOREGROUND:-0}" == "1" ]]; then
    # Run in foreground: stream to both terminal and log file
    exec python3 -u "${SCRIPT_DIR}/fuzzer.py" "$@" 2>&1 | tee "${LOG_FILE}"
else
    # Run detached with nohup; output only to log file
    nohup python3 -u "${SCRIPT_DIR}/fuzzer.py" "$@" > "${LOG_FILE}" 2>&1 &
    FUZZ_PID=$!
    echo "${FUZZ_PID}" > "${PID_FILE}"
    disown "${FUZZ_PID}" 2>/dev/null || true
    echo "[fuzzer.sh] started in background, pid=${FUZZ_PID}"
    echo "[fuzzer.sh] pid_file    = ${PID_FILE}"
    echo "[fuzzer.sh] tail log:   tail -f ${LOG_FILE}"
    echo "[fuzzer.sh] stop:       kill \$(cat ${PID_FILE})"
fi
