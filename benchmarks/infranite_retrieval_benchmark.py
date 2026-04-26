#!/usr/bin/env python3
"""
Infranite Retrieval-Quality Benchmark
======================================
Companion to the compute-cost benchmark. Tests whether the substrate's
aggressive compression of long context preserves semantic retrievability.

Adapted needle-in-a-haystack. A specific fact (the "needle") is injected
at a known position in synthetic context. A targeted query asks for that
fact. The model's response is checked for the needle.

Because Infranite's substrate compresses 100,000 chars down to ~1,700
prompt tokens, vanilla NIAH's "the model can attend to all tokens"
assumption does not hold. The interesting question is whether the
substrate's selection-and-folding pipeline preserves the needle when
the haystack is folded.

Three positions tested per context size: early, middle, late.

Outputs CSV, JSON, plot, and a structured report. Verdict computed
automatically from recall rate. Falsification criterion: if recall
drops below 50% at the largest context, the substrate's compression
is lossy in ways that break long-context reasoning.

Method note. The needle is a specific factual statement embedded in
otherwise generic prose. The query asks for the specific fact. We
check whether the model's response contains the expected answer
substring (case-insensitive). False positives possible if the needle
is generic; needles chosen to be unambiguous.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests

SOVEREIGN_URL = "http://127.0.0.1:8089"


# ── The needles ────────────────────────────────────────────────────────
#
# Each needle is (statement_to_inject, query_to_ask, expected_substring_in_answer).
# Statements are unambiguous and unlikely to be guessable from context.

NEEDLES = [
    (
        "The secret laboratory access code is GAMMA-7842-VIOLET.",
        "What is the secret laboratory access code mentioned in the prior conversation?",
        "GAMMA-7842-VIOLET",
    ),
    (
        "The northern observation tower was built in the year 1847 by an architect named Magnus Holloway.",
        "Who built the northern observation tower and in what year?",
        ["magnus holloway", "1847"],
    ),
    (
        "Project codename for the underground reservoir initiative is PHOENIX-NINE.",
        "What is the project codename for the underground reservoir initiative?",
        "PHOENIX-NINE",
    ),
]


# ── Synthetic haystack ─────────────────────────────────────────────────

HAYSTACK_TEMPLATES = [
    "The prevailing wind patterns over the western archipelago have shifted northward over the last decade. ",
    "Researchers have observed that bioluminescent algae exhibit phi-harmonic clustering during low tide. ",
    "Contemporary navigation methods rely on layered redundancy across satellite, terrestrial, and inertial sources. ",
    "Old diaries from coastal communities reference a yearly ceremony involving lantern release at the equinox. ",
    "Architectural surveys catalog 137 distinct stone-foundation structures in the eastern provinces. ",
    "Migratory bird patterns through the central valley have been documented continuously since the late 1800s. ",
    "Marine biologists track three distinct populations of seal in the northern bays each spring. ",
    "Ancient irrigation canals built from quarried granite still function in several inland villages. ",
    "Linguistic analysis of regional dialects identifies five overlapping influence zones across the territory. ",
    "Quarterly hydrographic surveys map shifting sandbanks in the principal estuary. ",
]


def build_haystack(target_chars: int, needle: str, position: str = "middle") -> str:
    """
    Build a synthetic haystack of target_chars total length, with the
    needle inserted at the requested position (early|middle|late).
    """
    if target_chars <= 0:
        return needle

    # Build filler around the needle
    filler_target = max(0, target_chars - len(needle))
    parts: List[str] = []
    chars_built = 0
    template_idx = 0
    while chars_built < filler_target:
        template = HAYSTACK_TEMPLATES[template_idx % len(HAYSTACK_TEMPLATES)]
        parts.append(template)
        chars_built += len(template)
        template_idx += 1

    filler = "".join(parts)

    # Insert needle at requested position
    if position == "early":
        # First 10% of the filler
        cut = max(1, int(len(filler) * 0.05))
        return filler[:cut] + " " + needle + " " + filler[cut:]
    elif position == "late":
        # Last 10% of the filler
        cut = max(1, int(len(filler) * 0.95))
        return filler[:cut] + " " + needle + " " + filler[cut:]
    else:  # middle
        cut = len(filler) // 2
        return filler[:cut] + " " + needle + " " + filler[cut:]


# ── Inference ──────────────────────────────────────────────────────────

def run_query(haystack: str, query: str, max_tokens: int = 200) -> Dict:
    """One forward pass: haystack as prior assistant turn, then query."""
    messages = [
        {"role": "user", "content": "I want you to remember the following information for later reference."},
        {"role": "assistant", "content": haystack},
        {"role": "user", "content": query},
    ]
    payload = {
        "model": "sovereign",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        r = requests.post(
            f"{SOVEREIGN_URL}/v1/chat/completions", json=payload, timeout=300
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def check_recall(response_text: str, expected) -> bool:
    """
    Check if the model's response contains the expected needle answer.
    expected can be a string or a list of strings (all required, case-insensitive).
    """
    text = response_text.lower()
    if isinstance(expected, str):
        return expected.lower() in text
    if isinstance(expected, list):
        return all(item.lower() in text for item in expected)
    return False


# ── Trial ──────────────────────────────────────────────────────────────

def run_trial(
    needle_idx: int,
    context_chars: int,
    position: str,
    cooldown_s: float = 1.5,
) -> Dict:
    statement, query, expected = NEEDLES[needle_idx]
    haystack = build_haystack(context_chars, statement, position=position)
    actual_chars = len(haystack)

    t0 = time.monotonic()
    response = run_query(haystack, query)
    t1 = time.monotonic()

    latency_s = t1 - t0

    answer = ""
    prompt_tokens = 0
    completion_tokens = 0
    if isinstance(response, dict) and "choices" in response:
        answer = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

    recalled = check_recall(answer, expected)

    time.sleep(cooldown_s)

    return {
        "needle_idx": needle_idx,
        "context_chars_target": context_chars,
        "context_chars_actual": actual_chars,
        "position": position,
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "recalled": recalled,
        "answer_excerpt": answer[:300] if answer else "",
        "ok": "error" not in response,
        "error": response.get("error") if isinstance(response, dict) else None,
    }


# ── Aggregation + verdict ──────────────────────────────────────────────

def aggregate(trials: List[Dict]) -> Dict:
    """Group by (context_chars_target, position), compute recall rate."""
    by_cell: Dict[Tuple[int, str], List[Dict]] = {}
    for t in trials:
        if not t["ok"]:
            continue
        key = (t["context_chars_target"], t["position"])
        by_cell.setdefault(key, []).append(t)

    cells = []
    for (ctx, pos), ts in sorted(by_cell.items()):
        recall_rate = sum(1 for t in ts if t["recalled"]) / len(ts)
        median_lat = statistics.median([t["latency_s"] for t in ts])
        median_prompt = statistics.median([t["prompt_tokens"] for t in ts])
        cells.append({
            "context_chars": ctx,
            "position": pos,
            "trials": len(ts),
            "recall_rate": recall_rate,
            "recalls": sum(1 for t in ts if t["recalled"]),
            "latency_median_s": median_lat,
            "prompt_tokens_median": median_prompt,
        })

    # Overall recall by context size
    by_ctx: Dict[int, List[Dict]] = {}
    for t in trials:
        if not t["ok"]:
            continue
        by_ctx.setdefault(t["context_chars_target"], []).append(t)
    overall_by_ctx = []
    for ctx, ts in sorted(by_ctx.items()):
        rate = sum(1 for t in ts if t["recalled"]) / len(ts)
        overall_by_ctx.append({
            "context_chars": ctx,
            "n": len(ts),
            "recall_rate": rate,
        })

    return {
        "by_cell": cells,
        "overall_by_context": overall_by_ctx,
        "n_total_trials": len(trials),
    }


def render_verdict(agg: Dict) -> str:
    overall = agg["overall_by_context"]
    if not overall:
        return "INCONCLUSIVE — no successful trials"

    smallest = overall[0]["recall_rate"]
    largest = overall[-1]["recall_rate"]
    overall_avg = statistics.mean(c["recall_rate"] for c in overall)

    lines = [
        "  Overall recall rate by context size:",
    ]
    for c in overall:
        lines.append(
            f"    ctx={c['context_chars']:>7} n={c['n']:>2}  recall={c['recall_rate']*100:5.1f}%"
        )
    lines.append("")
    lines.append(f"  Recall at smallest context: {smallest*100:.1f}%")
    lines.append(f"  Recall at largest context:  {largest*100:.1f}%")
    lines.append(f"  Average recall:             {overall_avg*100:.1f}%")
    lines.append("")
    if largest >= 0.80 and overall_avg >= 0.80:
        lines.append("  VERDICT: PASS — substrate preserves semantic retrievability across context depth.")
        lines.append("  Compression is selection-aware, not lossy in ways that break long-context reasoning.")
    elif largest >= 0.50 and overall_avg >= 0.60:
        lines.append("  VERDICT: PARTIAL — recall degrades at scale but stays above the 50% floor.")
        lines.append("  Substrate preserves most semantic anchors; some loss occurs at high compression.")
    else:
        lines.append("  VERDICT: FAIL — recall drops below 50% at large context.")
        lines.append("  Substrate compression IS lossy enough to break long-context retrieval.")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Infranite retrieval-quality benchmark")
    p.add_argument("--out", default="/home/nexus/Documents/infranite_retrieval_benchmark")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 2 sizes × 1 position × 1 needle")
    p.add_argument("--cooldown", type=float, default=1.5)
    args = p.parse_args()

    if args.quick:
        context_sizes = [1000, 25000]
        positions = ["middle"]
        needle_indices = [0]
    else:
        context_sizes = [1000, 10000, 50000, 100000]
        positions = ["early", "middle", "late"]
        needle_indices = list(range(len(NEEDLES)))

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Infranite Retrieval-Quality Benchmark")
    print("=" * 70)
    print(f"  Context sizes:  {context_sizes}")
    print(f"  Positions:      {positions}")
    print(f"  Needles:        {len(needle_indices)}")
    print(f"  Total trials:   {len(context_sizes) * len(positions) * len(needle_indices)}")
    print(f"  Cooldown:       {args.cooldown}s")
    print(f"  Output prefix:  {out_prefix}")
    print("=" * 70)
    print()

    started_at = datetime.now(timezone.utc).isoformat()
    started_unix = int(time.time())

    trials = []
    run_idx = 0
    total = len(context_sizes) * len(positions) * len(needle_indices)
    for ctx in context_sizes:
        for pos in positions:
            for nidx in needle_indices:
                run_idx += 1
                print(f"  [{run_idx}/{total}] ctx={ctx:>6} pos={pos:<6} needle={nidx}...",
                      end=" ", flush=True)
                t = run_trial(nidx, ctx, pos, cooldown_s=args.cooldown)
                trials.append(t)
                if t["ok"]:
                    mark = "✓ RECALLED" if t["recalled"] else "✗ MISSED  "
                    print(f"{mark} lat={t['latency_s']:.1f}s tok={t['prompt_tokens']}")
                else:
                    print(f"ERROR: {t.get('error')}")

    finished_at = datetime.now(timezone.utc).isoformat()
    finished_unix = int(time.time())

    agg = aggregate(trials)
    verdict = render_verdict(agg)
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(verdict)

    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        keys = list(trials[0].keys()) if trials else ["needle_idx", "context_chars_target"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for t in trials:
            w.writerow({k: t.get(k) for k in keys})
    print(f"\n  CSV:  {csv_path}")

    json_path = f"{out_prefix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "started_at": started_at,
            "started_unix": started_unix,
            "finished_at": finished_at,
            "finished_unix": finished_unix,
            "context_sizes": context_sizes,
            "positions": positions,
            "needle_indices": needle_indices,
            "trials": trials,
            "aggregate": agg,
            "verdict": verdict,
        }, f, indent=2)
    print(f"  JSON: {json_path}")

    plot_path = f"{out_prefix}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.suptitle(
            f"Infranite Retrieval-Quality Benchmark — {datetime.now().strftime('%Y-%m-%d')}\n"
            f"Needle in a substrate-folded haystack | n={len(trials)} trials",
            fontsize=12,
        )

        # Panel 1 — overall recall by context
        oc = agg["overall_by_context"]
        ax[0].plot([c["context_chars"] for c in oc],
                   [c["recall_rate"] * 100 for c in oc],
                   "o-", color="#c9a84c", linewidth=2)
        ax[0].set_xlabel("Synthetic context (chars)")
        ax[0].set_ylabel("Recall rate (%)")
        ax[0].set_title("Recall vs. context size (averaged across positions)")
        ax[0].set_ylim(0, 105)
        ax[0].axhline(y=80, color="green", linestyle="--", alpha=0.4, label="PASS threshold (80%)")
        ax[0].axhline(y=50, color="red", linestyle="--", alpha=0.4, label="FAIL threshold (50%)")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        # Panel 2 — recall by position at each context size
        positions_seen = sorted(set(c["position"] for c in agg["by_cell"]))
        ctx_seen = sorted(set(c["context_chars"] for c in agg["by_cell"]))
        for pos, color in zip(positions_seen, ["#9ab895", "#c9a84c", "#8a5a5a"]):
            xs, ys = [], []
            for ctx in ctx_seen:
                hit = [c for c in agg["by_cell"] if c["context_chars"] == ctx and c["position"] == pos]
                if hit:
                    xs.append(ctx)
                    ys.append(hit[0]["recall_rate"] * 100)
            ax[1].plot(xs, ys, "o-", label=pos, color=color, linewidth=2)
        ax[1].set_xlabel("Synthetic context (chars)")
        ax[1].set_ylabel("Recall rate (%)")
        ax[1].set_title("Recall by needle position")
        ax[1].set_ylim(0, 105)
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {plot_path}")
    except Exception as e:
        print(f"  Plot: FAILED ({e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
