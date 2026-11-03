"""Generate an NPU trace: one sequential read of a weight matrix.

The NPU reads the whole matrix in address order, 32 B per request. The matrix
is placed at the start of one layout's region in configs/hbm2e_pfm_regions.yaml,
so the reads are decoded with that layout's address mapping.

Example, 4096 x 4096 INT8 in the npu_friendly region (base 0x0):

  matrix size = 4096 * 4096 * 1 B = 16 MB
  requests    = 16 MB / 32 B = 524288
  trace       = LD 0x0, LD 0x20, LD 0x40, ... LD 0xffffe0
"""

import argparse
from pathlib import Path

import yaml

from mapper.model import ELEMENT_BYTES, REQUEST_BYTES
from scripts.make_region_config import OUT_PATH as REGION_CONFIG_PATH

TRACE_DIR = Path(__file__).resolve().parent


def read_regions():
    """Return region name -> (base, size) from the generated Ramulator config."""
    with open(REGION_CONFIG_PATH) as f:
        regions = yaml.safe_load(f)["MemorySystem"]["AddrMapper"]["regions"]
    return {region["name"]: (region["base"], region["size"]) for region in regions}


def parse_shape(text):
    """Turn "4096x14336" into (4096, 14336)."""
    rows, cols = text.lower().split("x")
    return int(rows), int(cols)


def matrix_bytes(shape):
    rows, cols = shape
    return rows * cols * ELEMENT_BYTES


def write_npu_trace(path, layout, shape):
    """Write one LD per 32 B of the matrix, in address order. Returns the request count."""
    base, size = read_regions()[layout]
    total = matrix_bytes(shape)
    if total > size:
        raise ValueError(f"{shape[0]}x{shape[1]} needs {total} B, region {layout} has {size} B")
    if total % REQUEST_BYTES:
        raise ValueError(f"matrix size {total} B is not a whole number of {REQUEST_BYTES} B requests")

    with open(path, "w") as f:
        for offset in range(0, total, REQUEST_BYTES):
            f.write(f"LD {base + offset:#x}\n")
    return total // REQUEST_BYTES


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", required=True, help="region name, e.g. npu_friendly")
    parser.add_argument("--shape", required=True, help="matrix shape, e.g. 4096x4096")
    parser.add_argument("-o", "--output", help="trace file (default traces/npu_<layout>_<shape>.trace)")
    args = parser.parse_args()

    output = args.output or TRACE_DIR / f"npu_{args.layout}_{args.shape}.trace"
    count = write_npu_trace(output, args.layout, parse_shape(args.shape))
    print(f"wrote {count} requests to {output}")


if __name__ == "__main__":
    main()
