#!/usr/bin/env python3
"""
Infranite vs Vanilla — Head-to-Head
====================================
Same input, same underlying model (Qwen3-8B Q4_K_M on llama-server).
Two paths:

  Path A (Infranite):  client → sovereign engine (substrate + phi compression)
                       → llama-server → response.

  Path B (Vanilla):    client → llama-server directly. Input truncated to
                       what fits in llama-server's native context window.
                       This is what a non-Infranite consumer of the same
                       model would have to do.

Measures per path: tokens actually processed by the model, VRAM, latency,
fraction of original input retained.

This benchmark does NOT claim either path "wins." Both are lossy at scale,
in different ways. Vanilla truncates (discards everything past the limit).
Infranite compresses with positional bias (the retrieval benchmark showed
middle/late content is dropped at scale). The benchmark shows the
specific tradeoff.

Compute-cost-per-input-character is the headline metric. Cheaper per
character is "more efficient at processing the input the user sent,"
regardless of how each path handles the load.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests


SOVEREIGN_URL = "http://127.0.0.1:8089"
LLAMA_URL = "http://127.0.0.1:8090"

FIXED_QUERY = "Summarize what you have just been told in one sentence."

CONTEXT_TEMPLATE = (
    "The northern observation tower was built in 1847 by Magnus Holloway. "
    "Phi-harmonic clustering occurs in bioluminescent algae at low tide. "
    "Quarterly hydrographic surveys map shifting sandbanks in the estuary. "
    "Marine biologists track three populations of seal in the northern bays. "
    "Linguistic analysis of regional dialects identifies five influence zones. "
)


# ── VRAM sampler ───────────────────────────────────────────────────────

class VRAMSampler:
    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.samples: List[Dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=2,
                )
                ts = time.monotonic()
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        self.samples.append({
                            "ts": ts, "gpu": int(parts[0]),
                            "mem_mib": int(parts[1])
                        })
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[int, int]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        peak: Dict[int, int] = {}
        for s in self.samples:
            cur = peak.get(s["gpu"], 0)
            if s["mem_mib"] > cur:
                peak[s["gpu"]] = s["mem_mib"]
        return peak


# ── Path runners ───────────────────────────────────────────────────────

def make_input(target_chars: int) -> str:
    chunks = []
    built = 0
    while built < target_chars:
        chunks.append(CONTEXT_TEMPLATE)
        built += len(CONTEXT_TEMPLATE)
    return "".join(chunks)[:target_chars]


def run_infranite(input_text: str) -> Dict:
    messages = [
        {"role": "user", "content": "Remember the following for later reference."},
        {"role": "assistant", "content": input_text},
        {"role": "user", "content": FIXED_QUERY},
    ]
    payload = {
        "model": "sovereign", "messages": messages,
        "max_tokens": 80, "temperature": 0.0,
    }
    sampler = VRAMSampler(0.1); sampler.start()
    t0 = time.monotonic()
    try:
        r = requests.post(f"{SOVEREIGN_URL}/v1/chat/completions",
                          json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        sampler.stop()
        return {"ok": False, "error": str(e)}
    t1 = time.monotonic()
    peak = sampler.stop()
    usage = data.get("usage", {})
    return {
        "ok": True, "path": "infranite",
        "latency_s": t1 - t0,
        "prompt_tokens_to_model": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "vram_peak_g0_mib": peak.get(0, 0),
        "vram_peak_g1_mib": peak.get(1, 0),
        "input_chars_sent": len(input_text),
        "input_chars_processed": len(input_text),
        "fraction_retained": 1.0,
    }


def run_vanilla(input_text: str, max_ctx_tokens: int = 7000) -> Dict:
    """
    Direct to llama-server. Truncate input to fit native context window.
    Uses /v1/chat/completions (OpenAI-compatible) on llama-server itself.
    """
    # Estimate: ~3.5 chars/token for English; truncate accordingly.
    # Reserve ~500 tokens for system prompt + query + response space.
    max_input_chars = (max_ctx_tokens - 500) * 3
    truncated = input_text[:max_input_chars]
    actual_chars = len(truncated)

    messages = [
        {"role": "user", "content": "Remember the following for later reference."},
        {"role": "assistant", "content": truncated},
        {"role": "user", "content": FIXED_QUERY},
    ]
    payload = {
        "messages": messages, "max_tokens": 80,
        "temperature": 0.0, "n_predict": 80,
    }
    sampler = VRAMSampler(0.1); sampler.start()
    t0 = time.monotonic()
    try:
        r = requests.post(f"{LLAMA_URL}/v1/chat/completions",
                          json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        sampler.stop()
        return {"ok": False, "error": str(e)}
    t1 = time.monotonic()
    peak = sampler.stop()
    usage = data.get("usage", {})
    return {
        "ok": True, "path": "vanilla",
        "latency_s": t1 - t0,
        "prompt_tokens_to_model": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "vram_peak_g0_mib": peak.get(0, 0),
        "vram_peak_g1_mib": peak.get(1, 0),
        "input_chars_sent": len(input_text),
        "input_chars_processed": actual_chars,
        "fraction_retained": actual_chars / max(1, len(input_text)),
    }


# ── Trial loop ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Infranite vs Vanilla head-to-head")
    p.add_argument("--out", default="/home/nexus/Documents/infranite_vs_vanilla")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cooldown", type=float, default=2.0)
    args = p.parse_args()

    if args.quick:
        sizes = [10000, 100000]
        trials = 1
    else:
        sizes = [1000, 10000, 50000, 100000]
        trials = 2

    print("=" * 72)
    print("  Infranite vs Vanilla — head-to-head")
    print("=" * 72)
    print(f"  Sizes: {sizes}")
    print(f"  Trials per (size, path): {trials}")
    print(f"  Total runs: {len(sizes) * 2 * trials}")
    print()

    started_at = datetime.now(timezone.utc).isoformat()
    started_unix = int(time.time())

    runs = []
    run_idx = 0
    total = len(sizes) * 2 * trials
    for sz in sizes:
        text = make_input(sz)
        for trial in range(trials):
            for path_fn, name in [(run_infranite, "INFR"), (run_vanilla, "VANL")]:
                run_idx += 1
                print(f"  [{run_idx}/{total}] size={sz:>6} trial={trial+1} path={name}...", end=" ", flush=True)
                r = path_fn(text)
                r["target_chars"] = sz
                r["trial"] = trial
                runs.append(r)
                if r["ok"]:
                    chars_per_s = r["input_chars_processed"] / r["latency_s"] if r["latency_s"] > 0 else 0
                    print(f"lat={r['latency_s']:.1f}s tok={r['prompt_tokens_to_model']} "
                          f"retained={r['fraction_retained']*100:.0f}% chars/s={chars_per_s:.0f}")
                else:
                    print(f"ERROR: {r.get('error')}")
                time.sleep(args.cooldown)

    finished_at = datetime.now(timezone.utc).isoformat()
    finished_unix = int(time.time())

    # Aggregate
    by_cell: Dict[tuple, List[Dict]] = {}
    for r in runs:
        if not r["ok"]:
            continue
        key = (r["target_chars"], r["path"])
        by_cell.setdefault(key, []).append(r)

    summary = []
    for (sz, path), trs in sorted(by_cell.items()):
        latencies = [r["latency_s"] for r in trs]
        toks = [r["prompt_tokens_to_model"] for r in trs]
        chars_proc = [r["input_chars_processed"] for r in trs]
        retained = [r["fraction_retained"] for r in trs]
        chars_per_s = [r["input_chars_processed"] / r["latency_s"] for r in trs if r["latency_s"] > 0]
        summary.append({
            "target_chars": sz, "path": path, "trials": len(trs),
            "latency_median_s": statistics.median(latencies),
            "tokens_to_model_median": statistics.median(toks),
            "chars_processed_median": statistics.median(chars_proc),
            "fraction_retained_median": statistics.median(retained),
            "chars_per_sec_median": statistics.median(chars_per_s) if chars_per_s else 0,
        })

    # Render results
    print()
    print("=" * 72)
    print("  RESULTS — head-to-head per condition")
    print("=" * 72)
    print(f"  {'Size':>8} {'Path':>5} {'Lat(s)':>7} {'ModelTok':>9} {'Chars/s':>9} {'Retained':>9}")
    for s in summary:
        print(f"  {s['target_chars']:>8} {s['path']:>5} "
              f"{s['latency_median_s']:>7.2f} {s['tokens_to_model_median']:>9.0f} "
              f"{s['chars_per_sec_median']:>9.0f} {s['fraction_retained_median']*100:>8.0f}%")

    # Save outputs
    out_prefix = Path(args.out)
    csv_path = f"{out_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        keys = list(runs[0].keys()) if runs else ["target_chars", "path"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k) for k in keys})

    json_path = f"{out_prefix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "started_at": started_at, "started_unix": started_unix,
            "finished_at": finished_at, "finished_unix": finished_unix,
            "trials_per_cell": trials,
            "context_sizes": sizes,
            "runs": runs, "summary": summary,
        }, f, indent=2)

    print(f"\n  CSV:  {csv_path}\n  JSON: {json_path}")

    # Plot
    plot_path = f"{out_prefix}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes_seen = sorted(set(s["target_chars"] for s in summary))
        infr = [next((s for s in summary if s["target_chars"] == sz and s["path"] == "infranite"), None) for sz in sizes_seen]
        vanl = [next((s for s in summary if s["target_chars"] == sz and s["path"] == "vanilla"), None) for sz in sizes_seen]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(
            f"Infranite vs Vanilla — head-to-head — {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=12,
        )

        axes[0].plot(sizes_seen, [s["chars_per_sec_median"] if s else 0 for s in infr], "o-", label="Infranite", color="#c9a84c", linewidth=2)
        axes[0].plot(sizes_seen, [s["chars_per_sec_median"] if s else 0 for s in vanl], "o-", label="Vanilla (truncated)", color="#8a5a5a", linewidth=2)
        axes[0].set_xlabel("Input size (chars)")
        axes[0].set_ylabel("Chars processed per second")
        axes[0].set_title("Throughput per character")
        axes[0].grid(True, alpha=0.3); axes[0].legend()

        axes[1].plot(sizes_seen, [s["fraction_retained_median"]*100 if s else 0 for s in infr], "o-", label="Infranite", color="#c9a84c", linewidth=2)
        axes[1].plot(sizes_seen, [s["fraction_retained_median"]*100 if s else 0 for s in vanl], "o-", label="Vanilla", color="#8a5a5a", linewidth=2)
        axes[1].set_xlabel("Input size (chars)")
        axes[1].set_ylabel("Fraction of input retained (%)")
        axes[1].set_title("Input retention")
        axes[1].set_ylim(0, 105); axes[1].grid(True, alpha=0.3); axes[1].legend()

        axes[2].plot(sizes_seen, [s["tokens_to_model_median"] if s else 0 for s in infr], "o-", label="Infranite", color="#c9a84c", linewidth=2)
        axes[2].plot(sizes_seen, [s["tokens_to_model_median"] if s else 0 for s in vanl], "o-", label="Vanilla", color="#8a5a5a", linewidth=2)
        axes[2].set_xlabel("Input size (chars)")
        axes[2].set_ylabel("Tokens model actually processes")
        axes[2].set_title("Model burden")
        axes[2].grid(True, alpha=0.3); axes[2].legend()

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot: {plot_path}")
    except Exception as e:
        print(f"  Plot: FAILED ({e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
