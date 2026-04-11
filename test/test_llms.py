#!/usr/bin/env python3
"""Smoke-test the two LLM endpoints used by fuzzer.sh.

LLMs covered:
  1. OpenAI-compatible chat completion  (eval/mutate role in fuzzer.py)
        env: LLM_API_BASE, LLM_API_KEY (or OPENAI_API_KEY), LLM_MODEL
  2. Anthropic Messages API             (Harbor's claude-code agent)
        env: ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, HARBOR_MODEL

Run from the Skillfuzz repo root after sourcing fuzzer.sh's env, or via
the companion test/test_llms.sh wrapper which exports the same defaults.

Exits 0 only if all probes succeed.
"""

from __future__ import annotations

import json
import os
import ssl
import sys
from typing import Any, Dict, Tuple
from urllib import error as urlerror
from urllib import request

TIMEOUT_SEC = int(os.environ.get("TEST_LLM_TIMEOUT", "60"))


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that actually has a trust store on conda Pythons.

    Resolution order:
      1. TEST_LLM_INSECURE=1            -> disable verification (diagnostic only)
      2. SSL_CERT_FILE env var          -> honored automatically by ssl
      3. certifi package if installed   -> use its CA bundle
      4. Common system CA bundle paths  -> Debian/Ubuntu, RHEL, Alpine, ...
      5. Python default                 -> may fail on conda
    """
    if os.environ.get("TEST_LLM_INSECURE") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("  [warn] TEST_LLM_INSECURE=1 — TLS verification DISABLED")
        return ctx

    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()

    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    for cand in (
        "/etc/ssl/certs/ca-certificates.crt",   # Debian, Ubuntu, Alpine
        "/etc/pki/tls/certs/ca-bundle.crt",     # RHEL, CentOS, Fedora
        "/etc/ssl/ca-bundle.pem",               # OpenSUSE
        "/etc/ssl/cert.pem",                    # macOS, FreeBSD
    ):
        if os.path.exists(cand):
            return ssl.create_default_context(cafile=cand)

    return ssl.create_default_context()


_SSL_CTX = _build_ssl_context()


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=TIMEOUT_SEC, context=_SSL_CTX) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, body
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def test_openai_compatible() -> bool:
    print("=" * 70)
    print("[1/2] OpenAI-compatible chat/completions  (eval/mutate LLM)")
    print("=" * 70)

    base = os.environ.get("LLM_API_BASE", "").rstrip("/")
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    print(f"  base    = {base or '(missing)'}")
    print(f"  model   = {model}")
    print(f"  key set = {'yes' if key else 'NO'}")

    if not base or not key:
        print("  -> SKIP: LLM_API_BASE or LLM_API_KEY not set")
        return False

    url = base + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with the single word: pong"},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    headers = {"Authorization": f"Bearer {key}"}

    status, body = _post_json(url, headers, payload)
    print(f"  HTTP    = {status}")
    if status != 200:
        print(f"  body    = {body[:500]}")
        print("  -> FAIL")
        return False

    try:
        obj = json.loads(body)
        content = obj["choices"][0]["message"]["content"]
        print(f"  reply   = {content!r}")
        print("  -> OK")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  parse error: {e}")
        print(f"  body    = {body[:500]}")
        print("  -> FAIL")
        return False


def test_anthropic_messages() -> bool:
    print("=" * 70)
    print("[2/2] Anthropic /v1/messages  (Harbor claude-code agent)")
    print("=" * 70)

    base = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("HARBOR_MODEL", "claude-haiku-4-5-20251001")

    print(f"  base    = {base or '(missing)'}")
    print(f"  model   = {model}")
    print(f"  key set = {'yes' if key else 'NO'}")

    if not base or not key:
        print("  -> SKIP: ANTHROPIC_BASE_URL or ANTHROPIC_API_KEY not set")
        return False

    url = base + "/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "Reply with the single word: pong"},
        ],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }

    status, body = _post_json(url, headers, payload)
    print(f"  HTTP    = {status}")
    if status != 200:
        print(f"  body    = {body[:500]}")
        print("  -> FAIL")
        return False

    try:
        obj = json.loads(body)
        # Standard Anthropic response: {"content": [{"type":"text","text":"..."}], ...}
        text = ""
        for blk in obj.get("content", []):
            if isinstance(blk, dict) and blk.get("type") == "text":
                text += blk.get("text", "")
        if not text and "choices" in obj:
            # Some gateways wrap as OpenAI-style; tolerate it.
            text = obj["choices"][0]["message"]["content"]
        print(f"  reply   = {text!r}")
        print("  -> OK")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  parse error: {e}")
        print(f"  body    = {body[:500]}")
        print("  -> FAIL")
        return False


def main() -> int:
    results = []
    results.append(("openai-compatible (eval/mutate)", test_openai_compatible()))
    print()
    results.append(("anthropic messages (claude-code)", test_anthropic_messages()))
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    all_ok = True
    for name, ok in results:
        flag = "OK  " if ok else "FAIL"
        print(f"  [{flag}] {name}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
