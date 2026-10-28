# unified_mem — PFM Dual-View Address Mapping (Group 13)

Course project implementing the address-mapping optimizer from
*Rethinking Unified Memory for NPU–PIM Systems: Dual-View Memory for Dynamic Inference of LLM* (PFM, arXiv 2608.06989).

## Goal

NPUs want a tensor spread across all memory channels; PIM units want it kept inside each bank.
We find one **jointly optimized** address mapping for a 2 MB superpage of a weight matrix, and compare it in
Ramulator 2.0 against an **NPU-friendly** and a **PIM-friendly** layout, using LLaMA3-8B layer shapes.

Reported metrics:

- Bandwidth utilization for sequential NPU reads
- Bandwidth utilization for PIM matrix-vector (GEMV) access

Out of scope: GPU kernel simulation, hardware synthesis, larger/MoE models.

## Tools

- [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) — cycle-level DRAM simulator (C++20, built on Linux/WSL)
- Python 3 + [PuLP](https://coin-or.github.io/pulp/) (CBC solver) — mapping optimizer

## Repository layout

| Path | Contents | Status |
| --- | --- | --- |
| `docs/PROJECT_DOC.md` | Specs, design decisions, implementation plan, results (what / why / how) | Active |
| `docs/VIVA_QA.md` | Every question asked during the project, with short viva-ready answers | Active |
| `mapper/` | PuLP mapping optimizer | Planned |
| `ramulator2/` | Ramulator 2.0 fork with per-region mapper and PIM request | Planned |
| `traces/` | NPU and PIM trace generators | Planned |
| `configs/` | HBM2e and region-mapping YAML configs | Planned |
| `results/` | Simulation outputs, tables and plots | Planned |

## Docs

The two docs also live online, where they can be edited and commented on:

- Project Doc: https://claude.ai/artifact/WGrKripYKdkqbHKokPDyhA
- Viva Q&A: https://claude.ai/artifact/XYKNfiq1E3fjaMVc62Md4e

The Markdown copies in `docs/` are synced from them whenever they change.

## Quick start

Coming once step 1 (the optimizer) lands. See the implementation plan in `docs/PROJECT_DOC.md`.
