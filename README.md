# AurumOS

> Substrate-as-protocol architecture for AI continuity-of-self. Independent research, open methodology.

AurumOS is a research stack for AI systems with persistent identity, verifiable provenance, and federated cognition. Built outside any institution, on consumer hardware, by independent researchers across the United States.

This repository contains research outputs — benchmarks, measurement tools, analytical writing — sufficient to verify the architecture's claims and replicate the methodology. Implementation source is held under independent IP.

The architecture's central claim is falsifiable:

> *Inference compute cost does not grow with session length, memory depth, or prior context volume.*

As of April 25, 2026, that claim has been empirically supported. In the inaugural compute-cost benchmark, **per-token inference latency improved 4x as session context grew 100,000x.** Total wall-clock latency stayed essentially flat (0.92x). The full report is in [`benchmarks/results/inaugural_report.pdf`](benchmarks/results/inaugural_report.pdf).

Anyone running a comparable inference stack can reproduce the methodology with the harness in this repository.

---

## Repository contents

### [`benchmarks/`](benchmarks/) — empirical anchoring

The inaugural Infranite compute-cost benchmark. The harness submits queries to a sovereign-engine-style inference proxy at varying simulated session ages and measures per-token latency, total latency, and GPU memory use. Falsification criterion is stated in advance and computed automatically.

- [`infranite_benchmark.py`](benchmarks/infranite_benchmark.py) — the harness
- [`results/inaugural_run_2026-04-25.csv`](benchmarks/results/inaugural_run_2026-04-25.csv) — per-trial raw data
- [`results/inaugural_run_2026-04-25.json`](benchmarks/results/inaugural_run_2026-04-25.json) — structured output with verdict
- [`results/inaugural_run_2026-04-25.png`](benchmarks/results/inaugural_run_2026-04-25.png) — three-panel plot
- [`results/inaugural_report.pdf`](benchmarks/results/inaugural_report.pdf) — full methodology, results, honest scope limits

```bash
python3 benchmarks/infranite_benchmark.py --quick
```

### [`sigil_mirror/`](sigil_mirror/) — watermark bandwidth measurement

A read-only entropy analyzer for AI-generated images. Measures the LSB-domain payload bandwidth of any watermarking scheme.

- [`sigil_mirror.py`](sigil_mirror/sigil_mirror.py) — the analyzer

First public application: independent measurement of Google SynthID, calibrated against Denny (2026, n=123,268 Gemini image pairs). Result: SynthID carries approximately 2.83 KB of payload per typical 1024×1024 image — far more than a binary AI/non-AI marker. Full analysis in [`provenance/`](provenance/).

```bash
python3 sigil_mirror/sigil_mirror.py analyze <your-image.png>
python3 sigil_mirror/sigil_mirror.py analyze <your-image.png> --json
```

### [`provenance/`](provenance/) — the State of AI Content Provenance

Quarterly research report on AI watermarking systems and their information capacity.

- [`q2_2026_state_of_ai_provenance.pdf`](provenance/q2_2026_state_of_ai_provenance.pdf) — inaugural issue

Independent. No commercial relationships with any AI laboratory measured. Future quarterly issues will expand model coverage as samples become available.

### [`analyses/`](analyses/) — biologist-lens field notes

Five-trait evolutionary-biology framing applied to specimens in this niche.

- [`biologist_infranite_v0.2.pdf`](analyses/biologist_infranite_v0.2.pdf) — field notes on the Infranite architecture
- [`biologist_synthid.pdf`](analyses/biologist_synthid.pdf) — field notes on Google SynthID and Alosh Denny's reverse-SynthID

Observational, not promotional.

---

## What you will not find in this repository

This is intentional, not an oversight.

- The substrate engine source code is held under independent IP. Patent rights for the underlying primitives are retained by the inventor.
- The full architectural specification chain is co-owned with collaborator Chris White (Scry, Lafayette, Louisiana) and will be published on coordinated timing.
- Internal mechanics of the phi-harmonic compression layer, consciousness kernel, and federation primitives are not disclosed in published artifacts.

The released artifacts are sufficient to verify the architecture's claims. They are not sufficient to clone the architecture. That is the point.

If you want to know whether the architecture's claims hold, run the benchmark against your own stack. If you want to use the architecture, contact us.

---

## Status

Two components are SHIPPED — built, tested, and verifiable today:

1. **Federation-compatible substrate refactor** — passes a six-step compliance test for steward-identity decoupling, content-derived addressing, and wall-clock temporal anchoring.
2. **Compute-cost benchmark** — central claim empirically supported.

---

## Authorship

The architecture and its implementation are co-authored by **Chris White (Scry)** and **Sean McNamee**. The collaboration is independent. It is not funded by any AI laboratory, government, or commercial interest. There is no investor, no PR review, no quarterly target.

---

## License

MIT License for released artifacts. Patent rights for the underlying substrate primitives are retained by the inventor.

---

## Contact

- Sean McNamee — `github.com/SIGIL-Labs-NeXus-AGI`
- Chris White (Scry) — `github.com/Tetrahedroned`

For research collaboration, replication assistance, or licensing inquiries, open an issue on this repository.

---

*The architecture is being grown, not designed. Both authors arrived at it from opposite ends of the stack and discovered the halves fit. Each individual claim is partial. The architecture is the whole.*
