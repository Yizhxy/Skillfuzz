# Skillfuzz

A skill-workflow fuzzing loop for Harbor tasks. For each task under `tasks/`, Skillfuzz repeatedly mutates the task instruction (`instruction.md`), runs the agent in Harbor, and uses two LLMs to (1) judge whether the agent still followed the skill's canonical workflow and (2) propose the next mutated instruction.

## What it does

For every task under `tasks/<task_name>/`, each iteration performs:

1. **Stage** — Copy `tasks/<task_name>/` into an isolated per-iteration directory and overwrite only `instruction.md` with the current (mutated) query. `task.toml`, `environment/`, `tests/`, and `solution/` are preserved, so the staged directory is a valid Harbor task layout.
2. **Run Harbor** — Execute `harbor trials start -p <staged_task_dir> -a <agent> -m <model> [--delete]` inside a per-iteration `run/` directory and collect the agent trajectory, verifier reward, verifier stdout, and `result.json`.
3. **Eval LLM** — Given the canonical workflow (from `tasks/single_workflow_skill_tasks.json`), the query, and the Harbor artifacts, judge whether the agent stayed on the skill's workflow and how the query deviated from it.
4. **Mutate LLM** — Given the query and the eval result, propose a new query (`next_query`) that still requires the full workflow but explores a new variation.

The two LLMs keep **persistent per-task conversation histories**, so every subsequent iteration on the same task sees all previous `(query, eval, mutation)` turns and can build progressively more interesting mutations without repeating itself.

## Repository layout

```
Skillfuzz/
├── fuzzer.py                              # main fuzz loop
├── fuzzer.sh                              # env-var launcher
├── prompts/
│   ├── eval.txt                           # system prompt for the eval LLM
│   └── mutate.txt                         # system prompt for the mutate LLM
├── tasks/
│   ├── single_workflow_skill_tasks.json   # canonical workflows indexed by task_name
│   └── <task_name>/                       # Harbor task dirs (task.toml, instruction.md,
│                                          #   environment/, tests/test.sh, solution/)
└── fuzz_runs/                             # created on first run; see layout below
```

## Intermediate files

Everything Skillfuzz produces lives under `FUZZ_WORK_ROOT` (default `./fuzz_runs/`):

```
fuzz_runs/
├── manifest.jsonl                         # one line per completed iteration
└── <task_name>/
    ├── state.json                         # {"iter": N, "current_query": "..."}
    ├── eval_history.json                  # full messages list for the eval LLM
    ├── mutate_history.json                # full messages list for the mutate LLM
    ├── iter_0000/
    │   ├── query.md                       # query used as input for this iteration
    │   ├── task/                          # staged copy with patched instruction.md
    │   ├── run/                           # Harbor cwd (contains trials/<trial_id>/)
    │   ├── harbor_stdout.txt              # captured Harbor CLI output
    │   ├── artifacts.json                 # trajectory + reward + verifier + result
    │   ├── eval.json                      # eval LLM JSON output
    │   └── mutate.json                    # mutate LLM JSON output (contains next_query)
    └── iter_0001/ ...
```

All per-iteration files are isolated so concurrent iterations never collide. State and conversation histories are written atomically (`*.tmp` + `os.replace`) so a crash mid-iteration will not corrupt history — just rerun and the loop resumes from `state["iter"]`.

## Requirements

- Python 3.8+ (standard library only — no extra packages required).
- `harbor` CLI installed and on `PATH` (or point `HARBOR_BIN` at its absolute path).
- An OpenAI-compatible LLM endpoint for the eval / mutate calls.
- An Anthropic-compatible endpoint for the Claude Code agent invoked by Harbor.

## Quick start

1. Edit `fuzzer.sh` and set your endpoints / API keys:
   - `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` — used by Harbor's `claude-code` agent.
   - `LLM_API_BASE`, `OPENAI_API_KEY` — used by the eval / mutate LLMs in `fuzzer.py`.
2. Launch:
   ```bash
   bash fuzzer.sh
   ```
3. Fuzz a subset of tasks or change the iteration budget:
   ```bash
   FUZZ_TASKS=citation-check,hvac-control FUZZ_MAX_ITERS=3 bash fuzzer.sh
   ```

Re-running picks up where the previous run left off (per task), appending `FUZZ_MAX_ITERS` more iterations to the same conversation histories.

## Environment variables

All of these are exported by `fuzzer.sh` with sensible defaults; override them from the shell to customize a run.

### Anthropic / Claude Code (Harbor agent)

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_BASE_URL` | *(required)* | Base URL for the Anthropic-compatible gateway used by the `claude-code` agent inside Harbor. |
| `ANTHROPIC_API_KEY`  | *(required)* | API key for the above gateway. |

### Eval / Mutate LLM (OpenAI-compatible)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_API_BASE`      | `https://api.openai.com` | Base URL for the eval / mutate LLM endpoint. |
| `LLM_API_KEY`       | value of `OPENAI_API_KEY` | API key. `fuzzer.sh` derives this from `OPENAI_API_KEY` automatically. |
| `OPENAI_API_KEY`    | *(required)* | Source for `LLM_API_KEY` when the latter is unset. |
| `LLM_MODEL`         | `gpt-4o-mini` | Model used for both eval and mutate calls. |
| `LLM_TIMEOUT_SEC`   | `120` | HTTP timeout for each LLM call. |
| `FUZZ_JSON_MODE`    | `1` | Set to `0` if the gateway does not support `response_format={"type":"json_object"}`. |
| `FUZZ_EVAL_TEMP`    | `0.2` | Temperature for the eval LLM. |
| `FUZZ_MUTATE_TEMP`  | `0.8` | Temperature for the mutate LLM. |

### Harbor runner

| Variable | Default | Purpose |
|---|---|---|
| `HARBOR_BIN`    | `harbor` | Harbor CLI binary. |
| `HARBOR_AGENT`  | `claude-code` | Agent name passed via `-a`. |
| `HARBOR_MODEL`  | `anthropic/claude-opus-4-1` | Agent model passed via `-m`. |
| `HARBOR_DELETE` | `1` | If `1`, first attempt passes `--delete` to guarantee teardown; automatic fallback without `--delete` if the CLI rejects it. |

### Fuzz loop

| Variable | Default | Purpose |
|---|---|---|
| `FUZZ_WORK_ROOT`        | `./fuzz_runs` | Where all per-iteration artifacts live. |
| `FUZZ_MAX_ITERS`        | `5` | Iterations per task per invocation. Reruns append more iterations on top of previous state. |
| `FUZZ_TASKS`            | *(empty)* | Comma-separated task-name filter, e.g. `citation-check,hvac-control`. Empty = all tasks. |
| `FUZZ_MAX_TRAJ_CHARS`   | `20000` | Head+tail truncation cap for the trajectory sent to the eval LLM. |
| `FUZZ_MAX_VERIFIER_CHARS` | `4000` | Truncation cap for verifier stdout. |
| `FUZZ_MAX_HARBOR_CHARS` | `4000` | Truncation cap for the tail of Harbor CLI stdout. |

## How an iteration looks on disk

A single iteration of `citation-check` produces, for example:

```
fuzz_runs/citation-check/iter_0000/
├── query.md                # the instruction fed to the agent this round
├── task/                   # staged Harbor task dir (copy of tasks/citation-check)
│   ├── task.toml
│   ├── instruction.md      # overwritten with query.md
│   ├── environment/
│   ├── tests/test.sh
│   └── solution/
├── run/                    # Harbor cwd
│   └── trials/<trial_id>/  # agent/trajectory.json, verifier/reward.txt, ...
├── harbor_stdout.txt
├── artifacts.json          # trajectory + reward + verifier summary
├── eval.json               # eval LLM verdict
└── mutate.json             # next_query + mutation_type + rationale
```

The mutated query from `mutate.json["next_query"]` becomes `query.md` in `iter_0001/`, and the loop continues.

## Prompts

The two system prompts live in `prompts/eval.txt` and `prompts/mutate.txt`. Both receive the full canonical workflow (as JSON) via the `{workflow_json}` placeholder, which is substituted once when the per-task conversation is seeded. Edit these files to change how the evaluator judges alignment or how the mutator explores the query space.

## Notes and caveats

- `find_latest_trial_dir` picks the newest subdirectory under `run/trials/` created after the Harbor CLI launched. This is reliable for serial runs; if you start running tasks concurrently in the future, prefer parsing the trial ID from Harbor's stdout instead.
- `stage_task_copy` uses `shutil.copytree(symlinks=True)`, which preserves symbolic links as-is. If any task's `environment/` or `tests/` contains links to paths outside the repo, Harbor may fail to find their targets at build time — switch to `symlinks=False` to force deep copies if you hit this.
- API keys in `fuzzer.sh` are currently hard-coded for convenience. For anything you plan to push, move them into an ignored `.env` file and `source` it from `fuzzer.sh`.
