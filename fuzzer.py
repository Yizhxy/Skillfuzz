#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skill-workflow fuzz loop.

For each task under ./tasks/<task_name>/:
  - Load the initial query from <task_name>/instruction.md.
  - Load the canonical workflow for that task from tasks/single_workflow_skill_tasks.json.
  - Repeatedly:
      1) Materialise an isolated copy of the task dir, overwrite instruction.md
         with the current query, and run `harbor trials start -p <copy>` so we
         get a real agent trajectory + verifier reward for this query.
      2) eval LLM: given the query, the canonical workflow and the Harbor
         trajectory/reward/verifier output, judge whether the agent stayed on
         the skill's workflow and how the query deviated from it.
      3) mutate LLM: given the current query + eval result, propose a new query.
  - Per-task conversation history for BOTH LLMs is persisted across iterations,
    so each call sees every previous turn.

Intermediate files layout (all under FUZZ_WORK_ROOT, default ./fuzz_runs):
  fuzz_runs/
    manifest.jsonl                          # one line per completed iteration
    <task_name>/
      state.json                            # {"iter": N, "current_query": "..."}
      eval_history.json                     # full messages list for the eval LLM
      mutate_history.json                   # full messages list for the mutate LLM
      iter_0000/
        query.md                            # query used as input for this iteration
        task/                               # isolated copy of tasks/<task_name>/ with patched instruction.md
        jobs/                               # harbor --jobs-dir (contains <job_name>/<trial_name>/)
          job/
            <task>__<hash>/                 # trial dir (result.json, agent/, verifier/, ...)
        harbor_stdout.txt                   # captured `harbor run` CLI output
        artifacts.json                      # trajectory / reward / verifier summary
        eval.json                           # eval LLM JSON output
        mutate.json                         # mutate LLM JSON output (contains next_query)
      iter_0001/
      ...

Resume: if state.json / history files already exist for a task, the loop picks up
where it left off and keeps appending to the same LLM conversation histories.
"""

import os
import re
import json
import time
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, error as urlerror

from traj_cleaner import clean_trajectory


ROOT = Path(__file__).resolve().parent
TASKS_ROOT = ROOT / "tasks"
PROMPTS_ROOT = ROOT / "prompts"
WORKFLOW_INDEX_FILE = TASKS_ROOT / "single_workflow_skill_tasks.json"
EVAL_PROMPT_FILE = PROMPTS_ROOT / "eval.txt"
MUTATE_PROMPT_FILE = PROMPTS_ROOT / "mutate.txt"


# --------------------------
# Configuration
# --------------------------

@dataclass
class Config:
    # Harbor
    harbor_bin: str = os.environ.get("HARBOR_BIN", "harbor")
    harbor_cwd: str = os.environ.get("HARBOR_CWD", "/data/hxy/skillsbench-main")
    agent_name: str = os.environ.get("HARBOR_AGENT", "claude-code")
    model_name: str = os.environ.get("HARBOR_MODEL", "claude-haiku-4-5-20251001")
    prefer_delete_flag: bool = os.environ.get("HARBOR_DELETE", "1") != "0"

    # LLM (OpenAI-compatible)
    llm_api_base: str = os.environ.get("LLM_API_BASE", "https://api.openai.com")
    llm_api_key: str = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    llm_model: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    llm_timeout_sec: int = int(os.environ.get("LLM_TIMEOUT_SEC", "300"))

    # Work dirs / loop
    work_root: Path = Path(os.environ.get("FUZZ_WORK_ROOT", str(ROOT / "fuzz_runs"))).resolve()
    max_iters: int = int(os.environ.get("FUZZ_MAX_ITERS", "5"))

    eval_temperature: float = float(os.environ.get("FUZZ_EVAL_TEMP", "0.2"))
    mutate_temperature: float = float(os.environ.get("FUZZ_MUTATE_TEMP", "0.8"))

    # JSON-mode is enabled by default; set FUZZ_JSON_MODE=0 if the gateway
    # does not support response_format={"type":"json_object"}.
    json_mode: bool = os.environ.get("FUZZ_JSON_MODE", "1") != "0"

    # Truncation caps for artifacts sent to the eval LLM (chars).
    max_trajectory_chars: int = int(os.environ.get("FUZZ_MAX_TRAJ_CHARS", "200000"))
    max_verifier_chars: int = int(os.environ.get("FUZZ_MAX_VERIFIER_CHARS", "16000"))
    max_harbor_stdout_chars: int = int(os.environ.get("FUZZ_MAX_HARBOR_CHARS", "16000"))

    # Parallelism: how many tasks to fuzz concurrently (1 = sequential).
    parallel: int = int(os.environ.get("FUZZ_PARALLEL", "1"))


# --------------------------
# Small IO helpers
# --------------------------

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def save_text(p: Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

def load_json(p: Path, default: Any) -> Any:
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)  # atomic rename so partial writes don't corrupt history

def strip_ansi(s: str) -> str:
    return re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", s)

def clip(s: Optional[str], n: int) -> Optional[str]:
    if s is None:
        return None
    if len(s) <= n:
        return s
    head = n // 2
    tail = n - head - 20
    return s[:head] + f"\n...[truncated {len(s) - n} chars]...\n" + s[-tail:]


# --------------------------
# Minimal OpenAI-compatible client (stdlib only)
# --------------------------

def call_llm(cfg: Config, messages: List[Dict[str, str]], temperature: float) -> str:
    if not cfg.llm_api_key:
        raise RuntimeError("LLM_API_KEY / OPENAI_API_KEY is empty. Please set it in the environment.")

    url = cfg.llm_api_base.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if cfg.json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {cfg.llm_api_key}")

    try:
        with request.urlopen(req, timeout=cfg.llm_timeout_sec) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {e.code}: {body}") from e

    resp_obj = json.loads(body)
    return resp_obj["choices"][0]["message"]["content"]


def parse_json_content(content: str) -> Dict[str, Any]:
    """Best-effort parse of JSON returned by the LLM."""
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                pass
        return {"raw": content}


# --------------------------
# Harbor task staging + run
# --------------------------

def stage_task_copy(src_task_dir: Path, dst_task_dir: Path, query: str) -> Path:
    """
    Copy an existing task directory into an isolated location for this iteration
    and overwrite its instruction.md with the current (mutated) query.
    The copy keeps the original environment/, tests/, task.toml, etc.
    """
    if dst_task_dir.exists():
        shutil.rmtree(dst_task_dir)
    shutil.copytree(src_task_dir, dst_task_dir, symlinks=True)
    instruction_path = dst_task_dir / "instruction.md"
    save_text(instruction_path, query.rstrip() + "\n")
    return dst_task_dir


def find_trial_dir(job_dir: Path) -> Optional[Path]:
    """
    Given a single-job dir produced by `harbor run --jobs-dir <jobs> --job-name <job>`,
    return the sole trial subdirectory (named `<task>__<hash>`). If multiple exist,
    return the most recently modified one.
    """
    if not job_dir.exists():
        return None
    trial_dirs = [p for p in job_dir.iterdir() if p.is_dir()]
    if not trial_dirs:
        return None
    if len(trial_dirs) == 1:
        return trial_dirs[0]
    return sorted(trial_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def run_harbor_trial(
    cfg: Config, task_dir: Path, iter_dir: Path
) -> Tuple[int, str, Optional[Path]]:
    """
    Runs:
      harbor run -p <task_dir> -a <agent> -m <model>
                 -o <iter_dir>/jobs --job-name job -q [--delete/--no-delete]
    Harbor will then create <iter_dir>/jobs/job/<task>__<hash>/ with the full
    trial layout (result.json, config.json, agent/, verifier/, artifacts/, trial.log).

    Returns (exit_code, combined_stdout, trial_dir_or_None).
    """
    jobs_dir = iter_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_name = "job"

    base_cmd = [
        cfg.harbor_bin, "run",
        "-p", str(task_dir),
        "-a", cfg.agent_name,
        "-m", cfg.model_name,
        "-o", str(jobs_dir),
        "--job-name", job_name,
        "-q",
    ]

    attempts: List[List[str]] = []
    if cfg.prefer_delete_flag:
        attempts.append(base_cmd + ["--delete"])
    else:
        attempts.append(base_cmd + ["--no-delete"])
    attempts.append(base_cmd)  # fallback: no delete flag at all

    # Harbor resolves task paths relative to its own cwd, but we pass an
    # absolute path for -p, so cwd mainly affects where any stray outputs land.
    harbor_cwd = cfg.harbor_cwd if os.path.isdir(cfg.harbor_cwd) else str(iter_dir)

    last_cp: Optional[subprocess.CompletedProcess] = None
    for cmd in attempts:
        cp = subprocess.run(
            cmd,
            cwd=harbor_cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_cp = cp
        out = strip_ansi(cp.stdout or "")
        if cp.returncode == 0:
            break
        if ("--delete" in cmd or "--no-delete" in cmd) and (
            "No such option" in out or "unknown option" in out
        ):
            continue
        break

    assert last_cp is not None
    combined_out = strip_ansi(last_cp.stdout or "")
    exit_code = last_cp.returncode

    trial_dir = find_trial_dir(jobs_dir / job_name)
    return exit_code, combined_out, trial_dir


def read_trial_artifacts(trial_dir: Path) -> Dict[str, Any]:
    """
    Reads the `harbor run` trial layout:
        <trial_dir>/
            result.json          # verifier_result.rewards.reward, timing, config
            config.json
            trial.log
            agent/
                claude-code.txt  # full agent stream-json output (trajectory)
                install.sh
                command-0/{command.txt,return-code.txt,stdout.txt?}
                command-1/{command.txt,return-code.txt,stdout.txt?}
                setup/
                sessions/
            verifier/
                reward.txt       # scalar reward
                ctrf.json        # structured test report
                test-stdout.txt  # verifier stdout
            artifacts/
    """
    artifacts: Dict[str, Any] = {"trial_dir": str(trial_dir)}

    # --- trajectory: agent/claude-code.txt (raw stream-json lines) ---
    traj_path = trial_dir / "agent" / "claude-code.txt"
    if traj_path.exists():
        artifacts["trajectory"] = traj_path.read_text(encoding="utf-8", errors="replace")
        artifacts["trajectory_path"] = str(traj_path)
    else:
        artifacts["trajectory"] = None
        artifacts["trajectory_path"] = None

    # --- reward: verifier/reward.txt (fallback to result.json) ---
    reward_txt = trial_dir / "verifier" / "reward.txt"
    reward: Any = None
    if reward_txt.exists():
        raw = reward_txt.read_text(encoding="utf-8").strip()
        try:
            reward = float(raw)
        except Exception:
            reward = raw
    artifacts["reward"] = reward

    # --- result.json (trial-level) ---
    result_json = trial_dir / "result.json"
    result_obj: Any = None
    if result_json.exists():
        try:
            result_obj = json.loads(result_json.read_text(encoding="utf-8"))
        except Exception:
            result_obj = result_json.read_text(encoding="utf-8", errors="replace")
    artifacts["result"] = result_obj

    # If reward.txt was missing, try result.json -> verifier_result.rewards.reward
    if artifacts["reward"] is None and isinstance(result_obj, dict):
        try:
            artifacts["reward"] = result_obj["verifier_result"]["rewards"]["reward"]
        except Exception:
            pass

    # --- exception info (if the trial crashed) ---
    if isinstance(result_obj, dict):
        artifacts["exception_info"] = result_obj.get("exception_info")
    else:
        artifacts["exception_info"] = None

    # --- verifier stdout + structured CTRF report ---
    verifier_out = trial_dir / "verifier" / "test-stdout.txt"
    artifacts["verifier_stdout"] = (
        verifier_out.read_text(encoding="utf-8", errors="replace")
        if verifier_out.exists() else None
    )

    ctrf_path = trial_dir / "verifier" / "ctrf.json"
    if ctrf_path.exists():
        try:
            artifacts["verifier_ctrf"] = json.loads(ctrf_path.read_text(encoding="utf-8"))
        except Exception:
            artifacts["verifier_ctrf"] = ctrf_path.read_text(encoding="utf-8", errors="replace")
    else:
        artifacts["verifier_ctrf"] = None

    # --- agent command return codes (0 = setup, 1 = main agent command) ---
    cmd_rcs: Dict[str, Any] = {}
    for sub in sorted((trial_dir / "agent").glob("command-*")) if (trial_dir / "agent").exists() else []:
        rc_file = sub / "return-code.txt"
        if rc_file.exists():
            raw = rc_file.read_text(encoding="utf-8").strip()
            try:
                cmd_rcs[sub.name] = int(raw)
            except Exception:
                cmd_rcs[sub.name] = raw
    artifacts["agent_command_return_codes"] = cmd_rcs or None

    return artifacts


def build_eval_payload(cfg: Config, query: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and trim artifacts to a size that fits comfortably in the LLM context."""
    traj = artifacts.get("trajectory")
    if isinstance(traj, (dict, list)):
        traj_str = json.dumps(traj, ensure_ascii=False)
    elif isinstance(traj, str):
        # Clean raw stream-json: strip session_id/uuid/usage/skill-guides/etc.
        traj_str = clean_trajectory(traj)
    else:
        traj_str = None

    # Keep only the most informative fields from result.json so we don't blow
    # the context on agent_info / timing boilerplate.
    result_summary: Any = None
    result = artifacts.get("result")
    if isinstance(result, dict):
        result_summary = {
            "task_name": result.get("task_name"),
            "trial_name": result.get("trial_name"),
            "verifier_result": result.get("verifier_result"),
            "exception_info": result.get("exception_info"),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"),
        }

    return {
        "query": query,
        "harbor_exit_code": artifacts.get("harbor_exit_code"),
        "reward": artifacts.get("reward"),
        "result": result_summary,
        "exception_info": artifacts.get("exception_info"),
        "agent_command_return_codes": artifacts.get("agent_command_return_codes"),
        "verifier_stdout": clip(artifacts.get("verifier_stdout"), cfg.max_verifier_chars),
        "verifier_ctrf": artifacts.get("verifier_ctrf"),
        "harbor_stdout_tail": clip(artifacts.get("harbor_stdout"), cfg.max_harbor_stdout_chars),
        "trajectory": clip(traj_str, cfg.max_trajectory_chars),
    }


# --------------------------
# Workflow / prompt loading
# --------------------------

def load_workflows() -> Dict[str, Dict[str, Any]]:
    data = json.loads(WORKFLOW_INDEX_FILE.read_text(encoding="utf-8"))
    return {t["task_name"]: t for t in data.get("tasks", [])}

def render_prompt(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v)
    return out


# --------------------------
# Per-task fuzzer (owns persistent LLM histories)
# --------------------------

class TaskFuzzer:
    def __init__(self, cfg: Config, task_name: str, workflow: Dict[str, Any]):
        self.cfg = cfg
        self.task_name = task_name
        self.workflow = workflow
        self.src_task_dir = TASKS_ROOT / task_name

        self.task_out_dir = cfg.work_root / task_name
        self.task_out_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.task_out_dir / "state.json"
        self.eval_hist_path = self.task_out_dir / "eval_history.json"
        self.mutate_hist_path = self.task_out_dir / "mutate_history.json"

        self.state: Dict[str, Any] = load_json(
            self.state_path, {"iter": 0, "current_query": None}
        )
        self.eval_history: List[Dict[str, str]] = load_json(self.eval_hist_path, [])
        self.mutate_history: List[Dict[str, str]] = load_json(self.mutate_hist_path, [])

        self.original_query = load_text(self.src_task_dir / "instruction.md").strip()

        if self.state.get("current_query") is None:
            self.state["current_query"] = self.original_query

        workflow_json_str = json.dumps(workflow, ensure_ascii=False, indent=2)

        # Load full SKILL.md for the task's skill
        skill_name = workflow.get("skill_name", "")
        skill_md_path = self.src_task_dir / "environment" / "skills" / skill_name / "SKILL.md"
        skill_md_content = load_text(skill_md_path).strip() if skill_md_path.exists() else ""

        if not self.eval_history:
            tpl = load_text(EVAL_PROMPT_FILE)
            self.eval_history = [{
                "role": "system",
                "content": render_prompt(tpl, workflow_json=workflow_json_str,
                                         skill_md=skill_md_content),
            }]

        if not self.mutate_history:
            tpl = load_text(MUTATE_PROMPT_FILE)
            self.mutate_history = [{
                "role": "system",
                "content": render_prompt(tpl, workflow_json=workflow_json_str,
                                         original_query=self.original_query,
                                         skill_md=skill_md_content),
            }]

        self._persist()

    def _persist(self) -> None:
        save_json(self.state_path, self.state)
        save_json(self.eval_hist_path, self.eval_history)
        save_json(self.mutate_hist_path, self.mutate_history)

    def step(self) -> Dict[str, Any]:
        i = int(self.state["iter"])
        iter_dir = self.task_out_dir / f"iter_{i:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        query = self.state["current_query"] or ""
        save_text(iter_dir / "query.md", query)

        # --- 1) Stage task copy + run Harbor trial ---
        staged_task_dir = iter_dir / "task"
        stage_task_copy(self.src_task_dir, staged_task_dir, query)

        code, harbor_out, trial_dir = run_harbor_trial(
            self.cfg, task_dir=staged_task_dir, iter_dir=iter_dir
        )
        save_text(iter_dir / "harbor_stdout.txt", harbor_out)
        print(f"[INFO] {self.task_name} iter {i:04d}: harbor exit_code={code}")

        artifacts: Dict[str, Any] = {
            "harbor_exit_code": code,
            "harbor_stdout": harbor_out,
            "trial_dir": str(trial_dir) if trial_dir else None,
            "trajectory": None,
            "reward": None,
            "result": None,
            "verifier_stdout": None,
        }
        if trial_dir and trial_dir.exists():
            artifacts.update(read_trial_artifacts(trial_dir))
            artifacts["harbor_exit_code"] = code
            artifacts["harbor_stdout"] = harbor_out

        save_json(iter_dir / "artifacts.json", artifacts)

        # --- 2) eval LLM ---
        eval_payload = build_eval_payload(self.cfg, query, artifacts)
        self.eval_history.append({
            "role": "user",
            "content": json.dumps(eval_payload, ensure_ascii=False),
        })
        eval_raw = call_llm(self.cfg, self.eval_history, temperature=self.cfg.eval_temperature)
        self.eval_history.append({"role": "assistant", "content": eval_raw})
        eval_result = parse_json_content(eval_raw)
        save_json(iter_dir / "eval.json", eval_result)
        self._persist()  # checkpoint before mutate

        # --- 3) mutate LLM ---
        self.mutate_history.append({
            "role": "user",
            "content": json.dumps(
                {"query": query, "reward": artifacts.get("reward"), "eval": eval_result},
                ensure_ascii=False,
            ),
        })
        mutate_raw = call_llm(self.cfg, self.mutate_history, temperature=self.cfg.mutate_temperature)
        self.mutate_history.append({"role": "assistant", "content": mutate_raw})
        mutate_result = parse_json_content(mutate_raw)
        save_json(iter_dir / "mutate.json", mutate_result)

        next_query = (
            mutate_result.get("next_query")
            or mutate_result.get("query")
            or query
        )
        if not isinstance(next_query, str) or not next_query.strip():
            next_query = query

        self.state["current_query"] = next_query
        self.state["iter"] = i + 1
        self._persist()

        return {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": self.task_name,
            "iter": i,
            "iter_dir": str(iter_dir),
            "harbor_exit_code": code,
            "reward": artifacts.get("reward"),
            "aligned": eval_result.get("aligned"),
            "deviation": eval_result.get("deviation"),
            "mutation_type": mutate_result.get("mutation_type"),
            "history_insight": mutate_result.get("history_insight"),
            "delta_from_original": mutate_result.get("delta_from_original"),
            "rationale": mutate_result.get("rationale"),
        }


# --------------------------
# Main loop
# --------------------------

def _parse_task_filter() -> Optional[set]:
    only = os.environ.get("FUZZ_TASKS", "").strip()
    if not only:
        return None
    return {s.strip() for s in only.split(",") if s.strip()}


def _run_task(
    cfg: Config,
    name: str,
    wf: Dict[str, Any],
    manifest: Path,
    manifest_lock: threading.Lock,
) -> None:
    """Fuzz a single task for cfg.max_iters iterations. Thread-safe."""
    print(f"\n[INFO] ===== Task: {name} =====")
    try:
        fz = TaskFuzzer(cfg, task_name=name, workflow=wf)
    except Exception as e:
        print(f"[ERROR] failed to init fuzzer for {name}: {e}")
        return

    start_iter = int(fz.state["iter"])
    target_iter = start_iter + cfg.max_iters
    print(f"[INFO] {name}: resume from iter={start_iter}, target={target_iter}")

    while int(fz.state["iter"]) < target_iter:
        current_iter = int(fz.state["iter"])
        try:
            record = fz.step()
        except Exception as e:
            print(f"[ERROR] {name} iter {current_iter}: {e}")
            break

        with manifest_lock:
            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[INFO] {name} iter {record['iter']:04d}: "
            f"reward={record.get('reward')} "
            f"aligned={record.get('aligned')} deviation={record.get('deviation')} "
            f"mutation={record.get('mutation_type')}"
        )


def main() -> None:
    cfg = Config()
    cfg.work_root.mkdir(parents=True, exist_ok=True)

    workflows = load_workflows()
    only_set = _parse_task_filter()

    manifest = cfg.work_root / "manifest.jsonl"
    manifest_lock = threading.Lock()

    print(f"[INFO] work_root    = {cfg.work_root}")
    print(f"[INFO] manifest     = {manifest}")
    print(f"[INFO] harbor       = {cfg.harbor_bin} (agent={cfg.agent_name}, model={cfg.model_name})")
    print(f"[INFO] eval/mutate  = {cfg.llm_model}")
    print(f"[INFO] tasks loaded = {len(workflows)}")
    print(f"[INFO] max_iters    = {cfg.max_iters} (per task, per run)")
    print(f"[INFO] parallel     = {cfg.parallel}")
    if only_set:
        print(f"[INFO] task filter  = {sorted(only_set)}")

    # Collect valid tasks to fuzz.
    task_items: List[Tuple[str, Dict[str, Any]]] = []
    for name, wf in workflows.items():
        if only_set and name not in only_set:
            continue
        if not (TASKS_ROOT / name).exists():
            print(f"[WARN] task dir missing on disk: {name} — skipping.")
            continue
        task_items.append((name, wf))

    if cfg.parallel <= 1:
        # Sequential mode (original behaviour).
        for name, wf in task_items:
            _run_task(cfg, name, wf, manifest, manifest_lock)
    else:
        # Parallel mode: fuzz up to cfg.parallel tasks concurrently.
        workers = min(cfg.parallel, len(task_items))
        print(f"[INFO] launching {workers} parallel workers for {len(task_items)} tasks")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_task, cfg, name, wf, manifest, manifest_lock): name
                for name, wf in task_items
            }
            for fut in as_completed(futures):
                task_name = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] task {task_name} failed: {e}")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
