# Unified Memory Address Mapping Optimizer

This project is implementation of following paper:

S. Zhao, L. Liu, T. Han, M. Wang, Y. Han, and Y. Wang, "Rethinking Unified Memory for NPU-PIM Systems: Dual-View Memory for Dynamic Inference of LLM," arXiv:2608.06989 [cs.AR], Aug. 2026. https://arxiv.org/abs/2608.06989


## About

NPUs read memory fastest when a tensor is spread across all channels. PIM units work best when data stays inside each bank. We solve for a single address mapping (per 2 MB superpage) that works well for both, and compare three layouts of a weight matrix in Ramulator 2.0:

- NPU-friendly
- PIM-friendly
- Jointly optimized

For each layout we measure bandwidth utilization for sequential NPU reads and for PIM matrix-vector (GEMV) access, using the layer shapes of LLaMA3-8B.

Not covered: GPU kernel simulation, hardware synthesis, larger models.

## Tools

- Ramulator 2.0 (https://github.com/CMU-SAFARI/ramulator2)
- Python 3 with PuLP (CBC solver)

## Structure

```
mapper/      mapping optimizer (PuLP)
ramulator2/  Ramulator 2.0 with per-region address mapping and PIM requests
traces/      NPU and PIM trace generation
configs/     HBM2e and mapping configs
scripts/     config generation, experiment runner, plots, mapper test
results/     outputs and plots
```

## Changes to Ramulator

All changes are on the `pfm` branch of the `ramulator2` submodule (based on tag `v2.0a`):

- `RegionMapper` address mapper: each memory region decodes addresses with its own bit layout.
- PIM-MAC request: one all-bank MAC reads the same row and column in all 16 banks of a pseudo-channel. Only bank timing applies (tRCD, tCCD_L, tRTP, tRP); it does not use the data bus. Trace lines are `PIM <addr>`.
- Generic controller: a bank is not precharged while a queued request still hits its open row.

## Setup

Tested on Ubuntu (WSL) with g++ 15 and CMake 4.

```bash
git clone --recursive https://github.com/yash9724/Unified_Memory.git
cd Unified_Memory

# Ramulator
cmake -S ramulator2 -B ~/build/ramulator2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build ~/build/ramulator2 -j

# Python
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Scripts look for Ramulator at `~/build/ramulator2/ramulator2`. Set `RAMULATOR_BIN` to use another path.

## Usage

Run everything from the repo root.

```bash
# 1. Solve for the joint layout and save all three layouts to mapper/out/mappings.json
.venv/bin/python -m mapper.save_mappings

# 2. Write configs/hbm2e_pfm_regions.yaml, one memory region per layout
.venv/bin/python -m scripts.make_region_config

# 3. Check the address decoding against hand-worked addresses
.venv/bin/python -m scripts.test_mapper

# 4. Run 4 shapes x 3 layouts x {NPU, PIM} through Ramulator (about 1 minute)
.venv/bin/python -m scripts.run_all

# 5. Figures and LaTeX tables
.venv/bin/python -m scripts.plot_results
```

Single traces can be made with `traces/npu_trace.py` and `traces/pim_trace.py`, for example:

```bash
.venv/bin/python -m traces.npu_trace --layout joint --shape 4096x4096
```

## Results

NPU utilization is measured against the 409.6 GB/s peak bandwidth. PIM utilization is measured against all banks finishing one MAC every tCCD_L (4369.1 GB/s inside the banks).

| Layout | NPU sequential read | PIM GEMV |
|---|---|---|
| NPU-friendly | 99.8 % | 81.0 - 86.2 % |
| PIM-friendly | 29.9 - 30.4 % | 81.0 - 86.2 % |
| Joint | 99.6 - 99.7 % | 81.0 - 86.2 % |

Per-shape numbers are in `results/results.csv`. Ramulator stops when the last request is sent, so the smallest shape (1024 x 4096) reads a few points high on PIM.

## Reference

S. Zhao, L. Liu, T. Han, M. Wang, Y. Han, and Y. Wang, "Rethinking Unified Memory for NPU-PIM Systems: Dual-View Memory for Dynamic Inference of LLM," arXiv:2608.06989 [cs.AR], Aug. 2026. https://arxiv.org/abs/2608.06989
