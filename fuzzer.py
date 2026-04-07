#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harbor external fuzz loop:
  query -> harbor trial (trace + verifier reward) -> eval(LLM) -> mutate(LLM) -> next query/tests -> ...

Key ideas:
- Each iteration generates a fresh Harbor task directory (instruction.md + task.toml + environment + tests/).
- Runs `harbor trials start -p <task_dir>` in an isolated run directory.
- Collects outputs (trajectory + reward + logs) from Harbor trial output.
- eval() and mutate() are implemented via LLM calls (OpenAI-compatible endpoint).
"""

import os
import re
import json
import time
import shutil
import uuid
import glob
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from urllib import request


# --------------------------
# Configuration
# --------------------------

@dataclass
class Config:
    # Harbor
    harbor_bin: str = os.environ.get("HARBOR_BIN", "harbor")
    agent_name: str = os.environ.get("HARBOR_AGENT", "claude-code")
    model_name: str = os.environ.get("HARBOR_MODEL", "anthropic/claude-opus-4-1")

    # Work dirs
    work_root: Path = Path(os.environ.get("FUZZ_WORK_ROOT", "./fuzz_runs")).resolve()
    keep_all_artifacts: bool = True   # keep all iteration dirs

    # Iterations
    max_iters: int = int(os.environ.get("FUZZ_MAX_ITERS", "10"))
    seed_query: str = os.environ.get("FUZZ_SEED_QUERY", "Solve the task described in the repository.")
    # Provide an initial minimal test script (will be replaced by mutate() later)
    seed_test_sh: str = os.environ.get(
        "FUZZ_SEED_TEST_SH",
        "#!/usr/bin/env bash\nset -euo pipefail\n# TODO: replace with real assertions\n# Always succeed for seed\necho 1 > /logs/verifier/reward.txt\n"
    )

    # LLM (OpenAI-compatible)
    llm_api_base: str = os.environ.get("LLM_API_BASE", "https://api.openai.com")
    llm_api_key: str = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    llm_timeout_sec: int = int(os.environ.get("LLM_TIMEOUT_SEC", "120"))

    # Isolation / deletion
    # We try to pass `--delete` first (if Harbor CLI supports it). If not supported, we retry without it.
    prefer_delete_flag: bool = True


# --------------------------
# Utilities
# --------------------------

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def sh(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

def strip_ansi(s: str) -> str:
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", s)


# --------------------------
# Minimal OpenAI-compatible LLM client (stdlib only)
# --------------------------

def call_openai_compat_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_sec: int = 120,
    temperature: float = 0.2,
    response_format_json: bool = True,
) -> Dict[str, Any]:
    """
    Calls OpenAI-compatible: POST {api_base}/v1/chat/completions
    Returns parsed JSON response dict.
    """
    if not api_key:
        raise RuntimeError("LLM_API_KEY/OPENAI_API_KEY is empty. Please set environment variable.")

    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # If the gateway supports response_format={"type":"json_object"}, enforce JSON
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

def extract_chat_content(resp: Dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp, ensure_ascii=False, indent=2)


# --------------------------
# Harbor task generation
# --------------------------

DEFAULT_MIN_DOCKERFILE = """\
FROM alpine:3.22
# bash is required for Harbor's docker environment
RUN apk add --no-cache bash coreutils
WORKDIR /app
"""

def make_task_toml(task_name: str) -> str:
    # Minimal viable task config; adjust timeouts/resources as needed.
    # The verifier reads /logs/verifier/reward.txt or reward.json as per Harbor task convention.
    return f'''\
schema_version = "1.1"

[task]
name = "{task_name}"
description = "Auto-generated fuzz task"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 1200.0

[environment]
cpus = 1
memory_mb = 2048
allow_internet = true
'''

def generate_task_dir(
    cfg: Config,
    iter_dir: Path,
    query: str,
    test_sh: str,
) -> Path:
    """
    Creates a Harbor task directory:
      task.toml, instruction.md, environment/Dockerfile, tests/test.sh
    """
    task_dir = iter_dir / "task"
    task_name = f"fuzz/{iter_dir.name}"

    safe_write(task_dir / "instruction.md", query + "\n")
    safe_write(task_dir / "task.toml", make_task_toml(task_name))
    safe_write(task_dir / "environment" / "Dockerfile", DEFAULT_MIN_DOCKERFILE)
    safe_write(task_dir / "tests" / "test.sh", test_sh if test_sh.endswith("\n") else test_sh + "\n")

    # Ensure test.sh executable on host; Harbor uploads it into container under /tests and executes it.
    os.chmod(task_dir / "tests" / "test.sh", 0o755)
    return task_dir


# --------------------------
# Run Harbor trial & collect outputs
# --------------------------

def find_latest_trial_dir(trials_root: Path, after_epoch: float) -> Optional:
    """
    Harbor writes trial outputs under a trials/ directory (default), each trial in its own subdir.
    We pick the newest directory created/modified after after_epoch.
    Output layout typically includes agent/trajectory.json, verifier/reward.txt, result.json, etc.
    """
    if not trials_root.exists():
        return None

    candidates = []
    for p in trials_root.iterdir():
        if p.is_dir():
            mtime = p.stat().st_mtime
            if mtime >= after_epoch - 1.0:  # slight slack
                candidates.append((mtime, p))
    if not candidates:
        # fallback: just take newest
        all_dirs = [(p.stat().st_mtime, p) for p in trials_root.iterdir() if p.is_dir()]
        if not all_dirs:
            return None
        return sorted(all_dirs, key=lambda x: x[0], reverse=True)[0][1]

    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def run_harbor_trial(cfg: Config, task_dir: Path, run_dir: Path) -> Tuple[int, str, Optional[Path]]:
    """
    Runs: harbor trials start -p <task_dir> -a <agent> -m <model> [--delete]
    in run_dir, so outputs are contained (e.g., run_dir/trials/...).
    Returns: (exit_code, combined_output, trial_output_dir)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        cfg.harbor_bin, "trials", "start",
        "-p", str(task_dir),
        "-a", cfg.agent_name,
        "-m", cfg.model_name,
    ]

    t0 = time.time()

    # Prefer delete to guarantee teardown (if CLI supports). If not supported, retry without.
    attempts = []
    if cfg.prefer_delete_flag:
        attempts.append(base_cmd + ["--delete"])
    attempts.append(base_cmd)

    last_cp = None
    for cmd in attempts:
        cp = sh(cmd, cwd=run_dir, env=os.environ.copy())
        last_cp = cp
        out = strip_ansi(cp.stdout or "")
        if cp.returncode == 0:
            break
        # If failure seems due to unknown option, try next.
        if "--delete" in cmd and ("No such option" in out or "unknown option" in out or "Error:" in out):
            continue
        # Otherwise, don't spam retries.
        break

    assert last_cp is not None
    combined_out = strip_ansi(last_cp.stdout or "")
    exit_code = last_cp.returncode

    # Try locate trial output directory
    trials_root = run_dir / "trials"
    trial_dir = find_latest_trial_dir(trials_root, after_epoch=t0)

    return exit_code, combined_out, trial_dir


def read_trial_artifacts(trial_dir: Path) -> Dict[str, Any]:
    """
    Reads common artifacts:
      agent/trajectory.json
      verifier/reward.txt or verifier/reward.json
      result.json (if exists)
      verifier/test-stdout.txt (if exists)
    """
    artifacts: Dict[str, Any] = {"trial_dir": str(trial_dir)}

    traj_path = trial_dir / "agent" / "trajectory.json"
    if traj_path.exists():
        artifacts["trajectory"] = json.loads(traj_path.read_text(encoding="utf-8"))
    else:
        artifacts["trajectory"] = None

    reward_txt = trial_dir / "verifier" / "reward.txt"
    reward_json = trial_dir / "verifier" / "reward.json"
    if reward_txt.exists():
        artifacts["reward"] = float(reward_txt.read_text(encoding="utf-8").strip())
    elif reward_json.exists():
        artifacts["reward"] = json.loads(reward_json.read_text(encoding="utf-8"))
    else:
        artifacts["reward"] = None

    result_json = trial_dir / "result.json"
    if result_json.exists():
        artifacts["result"] = json.loads(result_json.read_text(encoding="utf-8"))
    else:
        artifacts["result"] = None

    verifier_out = trial_dir / "verifier" / "test-stdout.txt"
    if verifier_out.exists():
        artifacts["verifier_stdout"] = verifier_out.read_text(encoding="utf-8", errors="replace")
    else:
        artifacts["verifier_stdout"] = None

    return artifacts


# --------------------------
# eval() and mutate() via LLM
# --------------------------

def eval_with_llm(cfg: Config, query: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses LLM to evaluate this iteration based on:
      - query
      - trace (trajectory)
      - test feedback (reward, verifier output)
    Returns a JSON dict (structure enforced by response_format if supported).
    """
    system = (
        "You are an evaluator for an agent fuzzing loop. "
        "Given the original query, the agent trajectory, and verifier feedback, "
        "produce a concise JSON evaluation object."
    )
    user = {
        "query": query,
        "reward": artifacts.get("reward"),
        "verifier_stdout": artifacts.get("verifier_stdout"),
        "trajectory": artifacts.get("trajectory"),
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        {"role": "user", "content": (
            "Return a JSON object with keys:\n"
            "- success: boolean\n"
            "- summary: string (what happened)\n"
            "- failure_modes: array of strings\n"
            "- signals: object (any useful numeric/text signals)\n"
            "- suggestions: array of strings (how to mutate query/tests)\n"
        )}
    ]

    resp = call_openai_compat_chat(
        api_base=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        messages=messages,
        timeout_sec=cfg.llm_timeout_sec,
        temperature=0.2,
        response_format_json=True,
    )
    content = extract_chat_content(resp)
    try:
        return json.loads(content)
    except Exception:
        # gateway may not support json_object; fallback to best-effort extraction
        return {"raw": content}


def mutate_with_llm(cfg: Config, prev_query: str, prev_test_sh: str, eval_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses LLM to mutate:
      - next_query
      - next_test_sh
    based on evaluation result.
    """
    system = (
        "You are a mutator in an agent fuzzing loop. "
        "Given previous query, previous test script, and the evaluation result, "
        "generate a new query and a new tests/test.sh. "
        "The new test script MUST write a numeric reward to /logs/verifier/reward.txt "
        "or /logs/verifier/reward.json."
    )

    user = {
        "prev_query": prev_query,
        "prev_test_sh": prev_test_sh,
        "eval": eval_result,
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        {"role": "user", "content": (
            "Return a JSON object with keys:\n"
            "- next_query: string\n"
            "- next_test_sh: string (a bash script; must end with writing reward)\n"
            "- rationale: string\n"
            "Constraints for next_test_sh:\n"
            "1) start with '#!/usr/bin/env bash'\n"
            "2) include 'set -euo pipefail'\n"
            "3) always write reward to /logs/verifier/reward.txt (float or int)\n"
        )}
    ]

    resp = call_openai_compat_chat(
        api_base=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        messages=messages,
        timeout_sec=cfg.llm_timeout_sec,
        temperature=0.7,
        response_format_json=True,
    )
    content = extract_chat_content(resp)
    try:
        out = json.loads(content)
    except Exception:
        out = {"raw": content}

    # Basic normalization / guardrails
    next_query = out.get("next_query", prev_query)
    next_test = out.get("next_test_sh", prev_test_sh)

    if "#!/usr/bin/env bash" not in next_test.splitlines()[0:2]:
        next_test = "#!/usr/bin/env bash\nset -euo pipefail\n" + next_test
    if "/logs/verifier/reward." not in next_test:
        # Ensure reward written, otherwise Harbor verifier has nothing to read.
        next_test = next_test.rstrip() + "\n\necho 0 > /logs/verifier/reward.txt\n"

    out["next_query"] = next_query
    out["next_test_sh"] = next_test
    return out


# --------------------------
# Main loop
# --------------------------

def main():
    cfg = Config()
    cfg.work_root.mkdir(parents=True, exist_ok=True)

    query = cfg.seed_query
    test_sh = cfg.seed_test_sh

    # Save a top-level manifest
    manifest_path = cfg.work_root / f"manifest_{now_ts()}_{uuid.uuid4().hex[:8]}.jsonl"
    print(f"[INFO] Work root: {cfg.work_root}")
    print(f"[INFO] Manifest: {manifest_path}")

    for i in range(cfg.max_iters):
        iter_id = f"iter_{i:04d}_{uuid.uuid4().hex[:6]}"
        iter_dir = cfg.work_root / iter_id
        run_dir = iter_dir / "run"
        print(f"\n[INFO] ===== Iteration {i}/{cfg.max_iters-1}: {iter_id} =====")

        # 1) Generate fresh task directory for this test case (isolated inputs)
        task_dir = generate_task_dir(cfg, iter_dir, query=query, test_sh=test_sh)

        # 2) Run Harbor trial (isolated environment per trial)
        code, out, trial_dir = run_harbor_trial(cfg, task_dir=task_dir, run_dir=run_dir)
        safe_write(iter_dir / "harbor_stdout.txt", out)
        print(f"[INFO] harbor exit_code={code}")
        if trial_dir:
            print(f"[INFO] trial_dir={trial_dir}")
        else:
            print("[WARN] Could not locate trial_dir automatically. Check run_dir/trials/.")

        # 3) Collect artifacts (trace + reward + logs)
        artifacts = {"harbor_exit_code": code, "harbor_stdout": out, "trial_dir": None}
        if trial_dir and trial_dir.exists():
            artifacts = read_trial_artifacts(trial_dir)
            artifacts["harbor_exit_code"] = code

        safe_write(iter_dir / "artifacts.json", json.dumps(artifacts, ensure_ascii=False, indent=2))

        # 4) eval(query, trace, reward, logs) -> eval_result (LLM)
        try:
            eval_result = eval_with_llm(cfg, query=query, artifacts=artifacts)
        except Exception as e:
            eval_result = {"success": False, "summary": f"eval failed: {e}", "failure_modes": ["eval_error"], "signals": {}, "suggestions": []}

        safe_write(iter_dir / "eval.json", json.dumps(eval_result, ensure_ascii=False, indent=2))
        print(f"[INFO] eval.success={eval_result.get('success')}")

        # 5) mutate(prev_query, prev_test_sh, eval_result) -> next_query, next_test_sh (LLM)
        try:
            mut = mutate_with_llm(cfg, prev_query=query, prev_test_sh=test_sh, eval_result=eval_result)
        except Exception as e:
            mut = {
                "next_query": query,
                "next_test_sh": test_sh,
                "rationale": f"mutate failed: {e}",
            }

        query = mut.get("next_query", query)
        test_sh = mut.get("next_test_sh", test_sh)

        safe_write(iter_dir / "mutate.json", json.dumps(mut, ensure_ascii=False, indent=2))

        # 6) Append to manifest (one line per iteration)
        record = {
            "iter": i,
            "iter_id": iter_id,
            "task_dir": str(task_dir),
            "run_dir": str(run_dir),
            "trial_dir": artifacts.get("trial_dir"),
            "reward": artifacts.get("reward"),
            "eval": eval_result,
            "mutate_rationale": mut.get("rationale"),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("[INFO] next query/test prepared.")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()