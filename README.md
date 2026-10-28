# unified_mem

Group 13, Memory Centric Computing project.

Implementation of the address mapping optimizer from the paper "Rethinking Unified Memory for NPU-PIM Systems: Dual-View Memory for Dynamic Inference of LLM" (PFM).

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
results/     outputs and plots
```

## Setup

To be added.
