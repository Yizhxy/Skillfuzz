#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory cleaner for Claude Code stream-json output (agent/claude-code.txt).

Reads the raw stream-json file, strips metadata noise (session_id, uuid, usage,
model boilerplate, fast_mode, etc.), and emits a compact representation that
keeps only the information an eval LLM needs:

  - Which tools the agent called, with what input
  - What results the tools returned
  - The agent's thinking and text reasoning
  - The final result summary

Usage as a library (from fuzzer.py):
    from traj_cleaner import clean_trajectory
    compact = clean_trajectory(raw_text, max_tool_result_chars=3000)

Usage as a CLI:
    python traj_cleaner.py agent/claude-code.txt              # prints to stdout
    python traj_cleaner.py agent/claude-code.txt -o clean.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Noise patterns to detect and replace with short summaries
# ---------------------------------------------------------------------------

# Skill guide: starts with "Base directory for this skill: ..." and contains
# multi-KB documentation that the agent received but is irrelevant to eval.
_SKILL_GUIDE_RE = re.compile(
    r"Base directory for this skill:\s*\S+[\s\S]*?ARGUMENTS:\s*(.*)",
    re.DOTALL,
)
_SKILL_GUIDE_PREFIX = "Base directory for this skill:"

# Persisted-output: large tool results saved to disk, only the summary matters.
_PERSISTED_OUTPUT_RE = re.compile(
    r"<persisted-output>\s*Output too large\s*\([^)]+\)\.\s*Full output saved to:\s*(\S+)",
)

# system-reminder tags injected by Claude Code harness — pure noise.
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>[\s\S]*?</system-reminder>",
)


def _strip_noise_from_text(text: str) -> str:
    """Remove or condense known noise patterns from a text string."""
    # Strip system-reminder blocks
    text = _SYSTEM_REMINDER_RE.sub("", text)

    # Condense skill guides: "Base directory ... <5KB docs> ... ARGUMENTS: /foo"
    # → "[skill guide loaded, args: /foo]"
    m = _SKILL_GUIDE_RE.search(text)
    if m:
        args = m.group(1).strip()
        text = f"[skill guide loaded, args: {args}]"
    elif _SKILL_GUIDE_PREFIX in text and len(text) > 2000:
        # Fallback: guide without ARGUMENTS line
        text = "[skill guide loaded]"

    # Condense persisted-output notices
    m = _PERSISTED_OUTPUT_RE.search(text)
    if m:
        text = f"[output too large, saved to: {m.group(1)}]"

    return text.strip()


# ---------------------------------------------------------------------------
# Per-content-block cleaners
# ---------------------------------------------------------------------------

def _clean_thinking(block: Dict[str, Any]) -> Optional[Dict[str, str]]:
    text = block.get("thinking", "").strip()
    if not text:
        return None
    return {"type": "thinking", "thinking": text}


def _clean_text(block: Dict[str, Any]) -> Optional[Dict[str, str]]:
    text = block.get("text", "").strip()
    if not text:
        return None
    text = _strip_noise_from_text(text)
    if not text:
        return None
    return {"type": "text", "text": text}


def _clean_tool_use(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "tool_use",
        "id": block.get("id", ""),
        "name": block.get("name", ""),
        "input": block.get("input", {}),
    }


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    head = n // 2
    tail = n - head - 40
    return s[:head] + f"\n...[trimmed {len(s) - n} chars]...\n" + s[-tail:]


def _clean_tool_result(
    block: Dict[str, Any], max_chars: int
) -> Dict[str, Any]:
    content = block.get("content", "")
    if isinstance(content, list):
        # Sometimes content is a list of text blocks
        content = "\n".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    if isinstance(content, str):
        content = _strip_noise_from_text(content)
        content = _clip(content, max_chars)
    return {
        "type": "tool_result",
        "tool_use_id": block.get("tool_use_id", ""),
        "content": content,
    }


# ---------------------------------------------------------------------------
# Per-line cleaners
# ---------------------------------------------------------------------------

def _clean_system_init(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the model name from the init line."""
    return {
        "type": "system",
        "subtype": "init",
        "model": obj.get("model", "unknown"),
    }


def _clean_assistant(
    obj: Dict[str, Any], max_tool_input_chars: int
) -> Optional[Dict[str, Any]]:
    msg = obj.get("message", {})
    contents = msg.get("content", [])
    cleaned: List[Dict[str, Any]] = []
    for block in contents:
        ct = block.get("type")
        if ct == "thinking":
            c = _clean_thinking(block)
            if c:
                cleaned.append(c)
        elif ct == "text":
            c = _clean_text(block)
            if c:
                cleaned.append(c)
        elif ct == "tool_use":
            c = _clean_tool_use(block)
            # Trim very large tool inputs (e.g. long Bash scripts)
            inp = c.get("input", {})
            if isinstance(inp, dict):
                for k, v in inp.items():
                    if isinstance(v, str) and len(v) > max_tool_input_chars:
                        inp[k] = _clip(v, max_tool_input_chars)
            cleaned.append(c)
    if not cleaned:
        return None
    return {"role": "assistant", "content": cleaned}


def _clean_user(
    obj: Dict[str, Any], max_tool_result_chars: int
) -> Optional[Dict[str, Any]]:
    msg = obj.get("message", {})
    contents = msg.get("content", [])
    cleaned: List[Dict[str, Any]] = []
    for block in contents:
        ct = block.get("type")
        if ct == "tool_result":
            cleaned.append(_clean_tool_result(block, max_tool_result_chars))
        elif ct == "text":
            c = _clean_text(block)
            if c:
                cleaned.append(c)
    if not cleaned:
        return None
    return {"role": "user", "content": cleaned}


def _clean_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "result",
        "num_turns": obj.get("num_turns"),
        "result": obj.get("result", ""),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_trajectory(
    raw_text: str,
    *,
    max_tool_result_chars: int = 3000,
    max_tool_input_chars: int = 4000,
) -> str:
    """
    Parse raw stream-json text from claude-code.txt, strip noise, return a
    compact JSON string (a JSON array of cleaned turns).

    Parameters
    ----------
    raw_text : str
        The full content of agent/claude-code.txt.
    max_tool_result_chars : int
        Trim individual tool result content beyond this length.
    max_tool_input_chars : int
        Trim individual tool use input string values beyond this length.

    Returns
    -------
    str
        A compact JSON array of cleaned turns.
    """
    turns: List[Dict[str, Any]] = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        line_type = obj.get("type", "")

        if line_type == "system" and obj.get("subtype") == "init":
            turns.append(_clean_system_init(obj))

        elif line_type == "assistant":
            cleaned = _clean_assistant(obj, max_tool_input_chars)
            if cleaned:
                turns.append(cleaned)

        elif line_type == "user":
            cleaned = _clean_user(obj, max_tool_result_chars)
            if cleaned:
                turns.append(cleaned)

        elif line_type == "result":
            turns.append(_clean_result(obj))
        # Skip everything else (e.g. system reminders, progress, etc.)

    return json.dumps(turns, ensure_ascii=False, indent=2)


def clean_trajectory_file(
    input_path: str,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Convenience: read a file, clean it, optionally write to output_path."""
    raw = Path(input_path).read_text(encoding="utf-8", errors="replace")
    cleaned = clean_trajectory(raw, **kwargs)
    if output_path:
        Path(output_path).write_text(cleaned, encoding="utf-8")
    return cleaned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean Claude Code stream-json trajectory")
    parser.add_argument("input", help="Path to agent/claude-code.txt")
    parser.add_argument("-o", "--output", help="Write cleaned JSON to this file")
    parser.add_argument("--max-tool-result", type=int, default=3000,
                        help="Max chars per tool result (default: 3000)")
    parser.add_argument("--max-tool-input", type=int, default=4000,
                        help="Max chars per tool input string (default: 4000)")
    args = parser.parse_args()

    cleaned = clean_trajectory_file(
        args.input,
        args.output,
        max_tool_result_chars=args.max_tool_result,
        max_tool_input_chars=args.max_tool_input,
    )
    if not args.output:
        print(cleaned)
    else:
        raw_size = Path(args.input).stat().st_size
        clean_size = len(cleaned.encode("utf-8"))
        print(f"Cleaned: {raw_size:,} -> {clean_size:,} bytes "
              f"({clean_size / raw_size * 100:.1f}%)")


if __name__ == "__main__":
    main()
