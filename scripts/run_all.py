"""Run every layout, shape and device through Ramulator and save the results.

For each LLaMA3-8B weight shape and each layout region, this writes an NPU
trace and a PIM trace, runs both through Ramulator with
configs/hbm2e_pfm_regions.yaml, and saves the utilization to
results/results.csv.

NPU utilization = data read / (cycles * tCK) / peak bandwidth, where peak is
409.6 GB/s (mapper/model.py, paper page 7).

PIM utilization = PIM-MACs * nCCDL / (pseudo-channels * cycles). At 100 %
every pseudo-channel finishes one all-bank MAC every nCCDL cycles. nCCDL is
tCCD_L rounded up to whole cycles, the same way Ramulator rounds it.

Run from the repo root:  .venv/bin/python -m scripts.run_all
"""

import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from mapper.model import ORG, PEAK_GBPS, REQUEST_BYTES, T_CCD_L, TIMING
from scripts.make_region_config import OUT_PATH as REGION_CONFIG_PATH
from traces.npu_trace import matrix_bytes, read_regions, write_npu_trace
from traces.pim_trace import write_pim_trace

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "results.csv"
DEFAULT_RAMULATOR = Path.home() / "build" / "ramulator2" / "ramulator2"

# LLaMA3-8B weight shapes (rows x cols). Only Dim 4096 is in the paper; the
# 1024 and 14336 sizes come from the public model config.
SHAPES = [(4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336)]

PSEUDO_CHANNELS = ORG["channel"] * ORG["pseudochannel"]
BANKS_PER_PSEUDO_CHANNEL = ORG["bankgroup"] * ORG["bank"]

# Ramulator's clock: tCK_ps = 1e6 / (rate / 2), with rate in MT/s.
TCK_PS = int(1e6 / (TIMING["rate"] / 2))


def to_cycles(t_ns):
    """Round a timing in ns to cycles exactly as Ramulator's JEDEC_rounding does."""
    t_ps = int(t_ns * 1000)
    return (t_ps * 1000 // TCK_PS + 974) // 1000


N_CCD_L = to_cycles(T_CCD_L)

# Peak PIM bandwidth inside the banks: every bank delivers 32 B per nCCDL.
PIM_PEAK_GBPS = PSEUDO_CHANNELS * BANKS_PER_PSEUDO_CHANNEL * REQUEST_BYTES / (N_CCD_L * TCK_PS / 1000)


def run_ramulator(trace_path, workdir):
    """Run one trace and return the memory cycles Ramulator reports."""
    with open(REGION_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["Frontend"]["path"] = str(trace_path)
    config_path = Path(workdir) / "run.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    # RAMULATOR_BIN lets the runs use a build somewhere else.
    ramulator = os.environ.get("RAMULATOR_BIN", DEFAULT_RAMULATOR)
    result = subprocess.run([str(ramulator), "-f", str(config_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ramulator failed on {trace_path}:\n{result.stderr}")

    match = re.search(r"memory_system_cycles:\s*(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"no cycle count in Ramulator output for {trace_path}")
    return int(match.group(1))


def npu_utilization(shape, cycles):
    """Return (GB/s, utilization) of reading the whole matrix in the given cycles."""
    gbps = matrix_bytes(shape) / (cycles * TCK_PS / 1000)   # bytes per ns = GB/s
    return gbps, gbps / PEAK_GBPS


def pim_utilization(macs, cycles):
    """Return (GB/s inside the banks, utilization) for the given PIM-MACs and cycles."""
    gbps = macs * BANKS_PER_PSEUDO_CHANNEL * REQUEST_BYTES / (cycles * TCK_PS / 1000)
    return gbps, gbps / PIM_PEAK_GBPS


def main():
    rows = []
    # Traces and configs are only needed for one run each, so they go in a temporary folder.
    with tempfile.TemporaryDirectory() as workdir:
        trace_path = Path(workdir) / "run.trace"
        for shape in SHAPES:
            name = f"{shape[0]}x{shape[1]}"
            for layout in read_regions():
                write_npu_trace(trace_path, layout, shape)
                cycles = run_ramulator(trace_path, workdir)
                gbps, util = npu_utilization(shape, cycles)
                rows.append({"shape": name, "layout": layout, "device": "NPU",
                             "cycles": cycles, "gbps": round(gbps, 2), "utilization": round(util, 4)})

                macs = write_pim_trace(trace_path, layout, shape)
                cycles = run_ramulator(trace_path, workdir)
                gbps, util = pim_utilization(macs, cycles)
                rows.append({"shape": name, "layout": layout, "device": "PIM",
                             "cycles": cycles, "gbps": round(gbps, 2), "utilization": round(util, 4)})

                print(f"{name:>11} {layout:<13} NPU {rows[-2]['utilization']:7.2%}  PIM {rows[-1]['utilization']:7.2%}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"NPU peak {PEAK_GBPS} GB/s, PIM peak {PIM_PEAK_GBPS:.1f} GB/s (nCCDL = {N_CCD_L} cycles)")
    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
