#!/usr/bin/env python3
"""
SIGIL Mirror v0.1 — Watermark Bandwidth Measurement
====================================================

Read-only entropy analyzer for AI-generated images.
Reports: addressable LSB capacity, per-channel anomaly rates,
estimated payload size, entropy class, free-LSB headroom.

This is v0.1: ANALYZE ONLY. No injection. No publish.
Working name "Mirror" is a placeholder pending NeXus's call.

Calibration baselines derived from Denny (2026) — empirical
LSB anomaly statistics across 123,268 Gemini image pairs:
  R: mean 1.35%, max 43.8%
  G: mean 0.40%, max 29.0%
  B: mean 0.46%, max 23.5%

Usage:
  sigil_mirror.py analyze <image>
  sigil_mirror.py analyze <image> --json
  sigil_mirror.py analyze <image> --calibration gemini-3.1
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Calibration constants — derived from Denny (2026) 123k-pair analysis
# ---------------------------------------------------------------------------

CALIBRATIONS = {
    "gemini-3.1": {
        "source": "Denny (2026), n=123,268 Gemini image pairs",
        "channel_anomaly_mean": {"R": 0.0135, "G": 0.0040, "B": 0.0046},
        "channel_anomaly_max":  {"R": 0.4382, "G": 0.2899, "B": 0.2353},
        "freq_diff_mean": 1.322,
        "freq_diff_std":  0.535,
        "indicators_2plus": 0.999,
        "notes": "Content-adaptive payload; faces ~1.7x stronger than backgrounds.",
    },
}

DEFAULT_CALIBRATION = "gemini-3.1"

# Anomaly threshold — fraction of LSB positions diverging from 0.5 baseline
# above which we flag a channel as "carrying payload"
ANOMALY_FLAG_THRESHOLD = 0.005   # 0.5% — well above noise floor
ENTROPY_UNIFORM_THRESHOLD = 0.999  # bits/lsb — near-uniform = encrypted/spread-spectrum

# Minimum image size for reliable detection (small images show high LSB
# entropy by statistical chance even without a watermark).
MIN_RELIABLE_PIXELS = 1024 * 1024


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def shannon_entropy(arr: np.ndarray) -> float:
    """Shannon entropy in bits per symbol."""
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def lsb_anomaly_rate(channel: np.ndarray) -> float:
    """
    Distance of LSB-ones-ratio from the 0.5 null baseline.
    A clean unmodified channel sits near 0.5; a watermarked one diverges.
    Returns absolute deviation (0.0 = no anomaly, 0.5 = maximally biased).
    """
    lsbs = (channel & 1).flatten()
    return float(abs(lsbs.mean() - 0.5))


def measure_image(path: str) -> dict:
    """
    Run full SIGIL Mirror analysis on a single image.
    Returns a result dict suitable for either pretty-printing or JSON.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 2:
        arr = arr[..., None]

    H, W, C = arr.shape
    total_lsb_bits = H * W * C
    file_size = os.path.getsize(path)

    channel_names = ["R", "G", "B", "A"][:C]

    # Per-channel measurements
    per_channel = {}
    for i, name in enumerate(channel_names):
        ch = arr[..., i]
        lsbs = (ch & 1).flatten()
        ones_ratio = float(lsbs.mean())
        anomaly = abs(ones_ratio - 0.5)
        ent = shannon_entropy(lsbs)
        # Run-flip rate (consecutive-bit independence proxy)
        flips = int(np.sum(np.diff(lsbs.astype(np.int8)) != 0))
        flip_rate = flips / max(1, lsbs.size - 1)
        per_channel[name] = {
            "lsb_ones_ratio": round(ones_ratio, 6),
            "anomaly_rate":   round(anomaly, 6),
            "entropy_bits_per_lsb": round(ent, 6),
            "flip_rate":      round(flip_rate, 6),
        }

    # Combined LSB plane entropy
    all_lsbs = np.concatenate([(arr[..., i] & 1).flatten() for i in range(C)])
    combined_entropy = shannon_entropy(all_lsbs)

    # Byte-level entropy of full image (upper bound on raw complexity)
    byte_entropy = shannon_entropy(arr.flatten())

    # Estimated payload from anomaly rates
    # Conservative estimate: sum of per-channel anomaly * channel-size
    # This counts only the "diverged" portion as payload, not the full LSB plane
    estimated_payload_bits = 0
    for i in range(C):
        ch_size = H * W
        anomaly = per_channel[channel_names[i]]["anomaly_rate"]
        # Multiply anomaly by 2 because deviation from 0.5 in either direction
        # represents the *biased* fraction of bits carrying signal
        estimated_payload_bits += int(ch_size * anomaly * 2)

    # Entropy classification
    if combined_entropy >= ENTROPY_UNIFORM_THRESHOLD:
        entropy_class = "uniform"
        entropy_interpretation = "encrypted or spread-spectrum encoding"
    elif combined_entropy >= 0.95:
        entropy_class = "near-uniform"
        entropy_interpretation = "lightly biased — possible structured payload"
    elif combined_entropy >= 0.80:
        entropy_class = "biased"
        entropy_interpretation = "structured content visible in LSB plane"
    else:
        entropy_class = "low-entropy"
        entropy_interpretation = "non-stego image (natural pixel correlations dominate)"

    # Channels flagged as anomalous
    flagged = [
        name for name, stats in per_channel.items()
        if stats["anomaly_rate"] >= ANOMALY_FLAG_THRESHOLD
    ]

    # Match against calibration. Two requirements per channel:
    #   (a) observed anomaly within calibrated range, AND
    #   (b) absolute anomaly above noise floor (small images can be uniform
    #       by chance with no watermark present).
    cal = CALIBRATIONS[DEFAULT_CALIBRATION]
    sig_match_score = 0
    image_pixels = H * W
    size_reliable = image_pixels >= MIN_RELIABLE_PIXELS
    for i, name in enumerate(channel_names):
        if name not in cal["channel_anomaly_mean"]:
            continue
        observed = per_channel[name]["anomaly_rate"]
        expected_mean = cal["channel_anomaly_mean"][name]
        expected_max = cal["channel_anomaly_max"][name]
        # Require observed >= 0.5x calibrated mean (not 0.25x) AND >= absolute
        # noise floor. Small image stats can hit "uniform" by chance.
        within_range = (expected_mean * 0.5) <= observed <= expected_max
        above_noise = observed >= ANOMALY_FLAG_THRESHOLD
        if within_range and above_noise:
            sig_match_score += 1

    # A real cryptographic watermark requires BOTH:
    #   (1) channel anomaly within calibrated SynthID range, AND
    #   (2) near-uniform LSB entropy (encrypted/spread-spectrum signature).
    # High anomaly + low entropy = natural image content (FLUX, JPEG, etc.),
    # not a watermark.
    is_uniform = combined_entropy >= ENTROPY_UNIFORM_THRESHOLD
    is_near_uniform = combined_entropy >= 0.99

    if sig_match_score >= 2 and is_uniform and size_reliable:
        signature_match = "SynthID-like (high confidence)"
    elif sig_match_score >= 2 and is_uniform and not size_reliable:
        signature_match = "SynthID-like (low confidence — image below 1MP, results unreliable)"
    elif sig_match_score >= 1 and is_near_uniform and size_reliable:
        signature_match = "SynthID-like (low confidence)"
    elif is_uniform and len(flagged) > 0 and size_reliable:
        signature_match = "uniform LSB anomaly — unknown watermark scheme"
    elif len(flagged) > 0 and not is_near_uniform:
        signature_match = "biased LSB plane — natural image content, no watermark signature"
    elif not size_reliable and is_uniform:
        signature_match = "inconclusive — image below 1MP minimum for reliable detection"
    else:
        signature_match = "no watermark signature detected"

    free_lsb_bits = total_lsb_bits - estimated_payload_bits
    free_lsb_fraction = free_lsb_bits / total_lsb_bits if total_lsb_bits else 0

    return {
        "file": path,
        "file_size_bytes": file_size,
        "format": im.format,
        "dimensions": {"width": W, "height": H, "channels": C, "mode": im.mode},
        "lsb_capacity": {
            "total_bits": total_lsb_bits,
            "total_bytes": total_lsb_bits // 8,
            "total_kb": round(total_lsb_bits / 8 / 1024, 2),
        },
        "per_channel": per_channel,
        "combined": {
            "lsb_entropy_bits_per_lsb": round(combined_entropy, 6),
            "byte_entropy_bits_per_byte": round(byte_entropy, 6),
            "entropy_class": entropy_class,
            "entropy_interpretation": entropy_interpretation,
        },
        "estimated_payload": {
            "bits": estimated_payload_bits,
            "bytes": estimated_payload_bits // 8,
            "kb": round(estimated_payload_bits / 8 / 1024, 4),
        },
        "free_lsb_headroom": {
            "bits": free_lsb_bits,
            "kb": round(free_lsb_bits / 8 / 1024, 2),
            "fraction": round(free_lsb_fraction, 4),
        },
        "flagged_channels": flagged,
        "signature_match": signature_match,
        "calibration_used": DEFAULT_CALIBRATION,
        "calibration_source": cal["source"],
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def render_human(result: dict) -> str:
    d = result["dimensions"]
    cap = result["lsb_capacity"]
    pay = result["estimated_payload"]
    free = result["free_lsb_headroom"]
    comb = result["combined"]

    lines = []
    lines.append("+" + "=" * 62 + "+")
    lines.append("|  SIGIL MIRROR v0.1 — WATERMARK BANDWIDTH REPORT             |")
    lines.append("+" + "=" * 62 + "+")
    lines.append(f"  File:   {result['file']}")
    lines.append(f"  Format: {result['format']}  {d['width']}x{d['height']} {d['mode']}")
    lines.append(f"  Size:   {result['file_size_bytes']:,} bytes on disk")
    lines.append("")
    lines.append(f"  LSB capacity:        {cap['total_kb']} KB ({cap['total_bits']:,} bits)")
    lines.append("")
    lines.append("  Per-channel anomaly:")
    for name, stats in result["per_channel"].items():
        flag = "  <-- flagged" if name in result["flagged_channels"] else ""
        lines.append(
            f"    {name} channel:  {stats['anomaly_rate']*100:6.3f}% LSB anomaly  "
            f"entropy={stats['entropy_bits_per_lsb']:.4f}{flag}"
        )
    lines.append("")
    lines.append(f"  Combined LSB entropy: {comb['lsb_entropy_bits_per_lsb']:.4f} bits/lsb")
    lines.append(f"  Entropy class:        {comb['entropy_class']}")
    lines.append(f"  Interpretation:       {comb['entropy_interpretation']}")
    lines.append("")
    lines.append(f"  Estimated payload:    {pay['kb']} KB ({pay['bits']:,} bits)")
    lines.append(f"  Free LSB headroom:    {free['kb']} KB ({free['fraction']*100:.1f}% of capacity)")
    lines.append("")
    lines.append(f"  Signature match:      {result['signature_match']}")
    lines.append(f"  Calibration:          {result['calibration_used']}")
    lines.append(f"  Source:               {result['calibration_source']}")
    lines.append("+" + "=" * 62 + "+")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        prog="sigil_mirror",
        description="SIGIL Mirror v0.1 — read-only watermark bandwidth analyzer.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Measure watermark bandwidth in an image.")
    a.add_argument("image", help="Path to the image file.")
    a.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable.")
    a.add_argument("--calibration", default=DEFAULT_CALIBRATION,
                   choices=list(CALIBRATIONS.keys()),
                   help="Calibration profile to compare against.")

    args = p.parse_args()

    if args.command == "analyze":
        try:
            result = measure_image(args.image)
        except FileNotFoundError as e:
            print(f"error: file not found: {e}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(render_human(result))


if __name__ == "__main__":
    main()
