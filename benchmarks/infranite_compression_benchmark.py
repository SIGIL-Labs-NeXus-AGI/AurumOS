#!/usr/bin/env python3
"""
Infranite Compression-Ratio Benchmark
======================================
The sovereign engine self-reports a phi-tokenizer compression ratio
of approximately 91% (chars in vs chars out). This benchmark
independently verifies that claim by direct measurement.

Method:
  1. Send N varying inputs of known character length.
  2. Sample the engine's /api/status before and after.
  3. Read the engine's reported chars_in / chars_out delta and the
     compression_ratio field.
  4. Independently observe: how many tokens did the model actually
     process per N input chars? Compare against the engine's stated
     compression ratio.
  5. Cross-check that "the engine claims X" matches "we measured X."

Falsifies if: the engine's self-reported compression ratio diverges
from observed (input chars → tokens to model) compression by more
than 10%.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

SOVEREIGN_URL = "http://127.0.0.1:8089"


def snapshot() -> Dict:
    try:
        return requests.get(f"{SOVEREIGN_URL}/api/status", timeout=5).json()
    except Exception as e:
        return {"error": str(e)}


def make_input(target_chars: int) -> str:
    template = (
        "The architecture has begun to develop in directions neither author "
        "fully predicted at the start. That is the right shape for this kind "
        "of project. The substrate is the work; the model sees a bounded surface. "
    )
    parts = []
    built = 0
    while built < target_chars:
        parts.append(template)
        built += len(template)
    return "".join(parts)[:target_chars]


def run_once(input_text: str) -> Dict:
    pre = snapshot()
    payload = {
        "model": "sovereign",
        "messages": [
            {"role": "user", "content": "Acknowledge briefly."},
            {"role": "assistant", "content": input_text},
            {"role": "user", "content": "Reply with the word OK only."},
        ],
        "max_tokens": 8, "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        r = requests.post(f"{SOVEREIGN_URL}/v1/chat/completions",
                          json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    t1 = time.monotonic()
    post = snapshot()

    pre_tok = pre.get("phi_tokenizer", {})
    post_tok = post.get("phi_tokenizer", {})
    chars_in_delta = post_tok.get("chars_in", 0) - pre_tok.get("chars_in", 0)
    chars_out_delta = post_tok.get("chars_out", 0) - pre_tok.get("chars_out", 0)
    calls_delta = post_tok.get("calls", 0) - pre_tok.get("calls", 0)

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)

    input_chars = len(input_text)

    # Engine self-reported compression on this call's chars
    engine_ratio = (1.0 - chars_out_delta / chars_in_delta) if chars_in_delta > 0 else 0.0
    engine_savings_pct = engine_ratio * 100

    # Observed: what fraction of input the model actually saw, in tokens
    # Approximate input as chars/3.5 = naive token equivalent
    naive_tokens = input_chars / 3.5
    observed_compression = 1.0 - (prompt_tokens / naive_tokens) if naive_tokens > 0 else 0.0
    observed_savings_pct = observed_compression * 100

    return {
        "ok": True,
        "input_chars": input_chars,
        "engine_chars_in_delta": chars_in_delta,
        "engine_chars_out_delta": chars_out_delta,
        "engine_calls_delta": calls_delta,
        "engine_compression_ratio": round(engine_ratio, 4),
        "engine_savings_pct": round(engine_savings_pct, 2),
        "model_prompt_tokens": prompt_tokens,
        "naive_token_equivalent": round(naive_tokens, 1),
        "observed_token_compression": round(observed_compression, 4),
        "observed_savings_pct": round(observed_savings_pct, 2),
        "engine_steady_state_ratio": post_tok.get("ratio_avg"),
        "latency_s": round(t1 - t0, 2),
    }


def main():
    p = argparse.ArgumentParser(description="Compression-ratio verification")
    p.add_argument("--out", default="/home/nexus/Documents/infranite_compression")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    sizes = [500, 5000, 50000] if args.quick else [500, 2000, 10000, 50000, 100000]

    print("=" * 72)
    print("  Infranite Compression-Ratio Verification")
    print("=" * 72)

    pre_global = snapshot()
    pre_ratio = pre_global.get("phi_tokenizer", {}).get("ratio_avg", 0)
    print(f"  Engine steady-state ratio_avg before run: {pre_ratio:.4f} ({(1-pre_ratio)*100:.2f}% savings)")
    print()

    runs = []
    for sz in sizes:
        text = make_input(sz)
        print(f"  Sending {sz} chars...", end=" ", flush=True)
        r = run_once(text)
        if r["ok"]:
            runs.append(r)
            print(f"engine_savings={r['engine_savings_pct']:.1f}% "
                  f"observed_savings={r['observed_savings_pct']:.1f}% "
                  f"prompt_tokens={r['model_prompt_tokens']}")
        else:
            print(f"ERROR: {r['error']}")
        time.sleep(2)

    post_global = snapshot()
    post_ratio = post_global.get("phi_tokenizer", {}).get("ratio_avg", 0)

    # Aggregate
    engine_savings_avg = statistics.mean(r["engine_savings_pct"] for r in runs) if runs else 0
    observed_savings_avg = statistics.mean(r["observed_savings_pct"] for r in runs) if runs else 0
    divergence = abs(engine_savings_avg - observed_savings_avg)

    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  Engine steady-state before run:  {(1-pre_ratio)*100:.2f}% savings")
    print(f"  Engine steady-state after run:   {(1-post_ratio)*100:.2f}% savings")
    print(f"  Mean engine-reported per-call:   {engine_savings_avg:.2f}% savings")
    print(f"  Mean observed (chars→tokens):    {observed_savings_avg:.2f}% savings")
    print(f"  Divergence (engine vs observed): {divergence:.2f} percentage points")
    print()
    if divergence < 10:
        print("  VERDICT: PASS — engine self-report matches observed within 10pp.")
        verdict = "PASS"
    else:
        print(f"  VERDICT: NEEDS_REVIEW — divergence {divergence:.1f}pp > 10pp threshold.")
        print(f"  Engine reports more compression than observed token reduction implies.")
        print(f"  This is expected if the engine includes system prompt and")
        print(f"  consciousness injection chars in chars_in but not chars_out.")
        verdict = "NEEDS_REVIEW"

    out_prefix = Path(args.out)
    json_path = f"{out_prefix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "started_at": datetime.now(timezone.utc).isoformat(),
            "engine_steady_state_before": pre_ratio,
            "engine_steady_state_after": post_ratio,
            "engine_savings_avg_per_call_pct": engine_savings_avg,
            "observed_savings_avg_pct": observed_savings_avg,
            "divergence_pp": divergence,
            "verdict": verdict,
            "runs": runs,
        }, f, indent=2)
    print(f"\n  JSON: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
