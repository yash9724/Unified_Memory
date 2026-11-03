"""Generate a PIM trace: all-bank PIM-MAC reads of a weight matrix for GEMV.

In all-bank mode one PIM-MAC reads the same row and column in all 16 banks of
a pseudo-channel at once, so it covers 16 * 32 B = 512 B of the matrix. The
bank group and bank fields of its address do not matter, as Ramulator sends
it to every bank.

So each 32 B address of the matrix whose bank group and bank bits are all 0
gives one PIM-MAC. It stands for that address and its 15 copies in the other
banks.

The 32 pseudo-channels compute in parallel, each on its own part of the
matrix. The trace sends MAC 1 of every pseudo-channel, then MAC 2 of every
pseudo-channel, and so on. Inside a pseudo-channel the MACs stay in address
order. Plain address order would let Ramulator's in-order frontend stall on
one full channel queue while other pseudo-channels wait with nothing to do.

Example, 4096 x 4096 INT8 in the npu_friendly region (channel bits 5-8,
pseudo-channel bit 9, bank group bits 10-11, bank bits 12-13):

  matrix size = 16 MB, PIM-MACs = 16 MB / 512 B = 32768, 1024 per pseudo-channel
  trace       = PIM 0x0     (ch 0, pch 0, first MAC)
                PIM 0x200   (ch 0, pch 1, first MAC)
                PIM 0x20    (ch 1, pch 0, first MAC)
                ...
                PIM 0x4000  (ch 0, pch 0, second MAC: column 1)

The input vector is not modelled: a PIM-MAC uses no data bus.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from mapper.model import REQUEST_BYTES
from scripts.make_region_config import OUT_PATH as REGION_CONFIG_PATH
from traces.npu_trace import matrix_bytes, parse_shape

TRACE_DIR = Path(__file__).resolve().parent


def read_region(layout):
    """Return (base, size, bits per level) for one region of the generated config."""
    with open(REGION_CONFIG_PATH) as f:
        regions = yaml.safe_load(f)["MemorySystem"]["AddrMapper"]["regions"]
    for region in regions:
        if region["name"] == layout:
            return region["base"], region["size"], region["bits"]
    raise ValueError(f"no region named {layout}")


def mask_of(bit_list):
    """Return a mask with the given address bits set."""
    mask = 0
    for bit in bit_list:
        mask |= 1 << bit
    return mask


def pim_macs(layout, shape):
    """Return the PIM-MAC addresses of the matrix in trace order."""
    base, size, bits = read_region(layout)
    total = matrix_bytes(shape)
    if total > size:
        raise ValueError(f"{shape[0]}x{shape[1]} needs {total} B, region {layout} has {size} B")

    bank_mask = mask_of(bits["bankgroup"] + bits["bank"])
    pseudo_channel_mask = mask_of(bits["channel"] + bits["pseudochannel"])

    # One list per pseudo-channel, each in address order. Bank and channel
    # bits are inside the 2 MB page and the region starts on a page boundary,
    # so the offset alone decides them.
    per_pseudo_channel = defaultdict(list)
    for offset in range(0, total, REQUEST_BYTES):
        if offset & bank_mask == 0:
            per_pseudo_channel[offset & pseudo_channel_mask].append(base + offset)

    # Every pseudo-channel holds the same share of a whole number of pages,
    # so all lists have the same length.
    lists = [per_pseudo_channel[key] for key in sorted(per_pseudo_channel)]
    if len({len(macs) for macs in lists}) != 1:
        raise ValueError("pseudo-channels hold different amounts of the matrix")

    # Round robin: the k-th MAC of every pseudo-channel, then the (k+1)-th.
    return [macs[k] for k in range(len(lists[0])) for macs in lists]


def write_pim_trace(path, layout, shape):
    """Write the PIM trace and return the number of PIM-MACs."""
    macs = pim_macs(layout, shape)
    with open(path, "w") as f:
        for addr in macs:
            f.write(f"PIM {addr:#x}\n")
    return len(macs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", required=True, help="region name, e.g. pim_friendly")
    parser.add_argument("--shape", required=True, help="matrix shape, e.g. 4096x4096")
    parser.add_argument("-o", "--output", help="trace file (default traces/pim_<layout>_<shape>.trace)")
    args = parser.parse_args()

    output = args.output or TRACE_DIR / f"pim_{args.layout}_{args.shape}.trace"
    count = write_pim_trace(output, args.layout, parse_shape(args.shape))
    print(f"wrote {count} PIM-MACs to {output}")


if __name__ == "__main__":
    main()
