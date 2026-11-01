"""
Shared definitions for address mapping optimizer.

Levels follow Fig. 9 of the paper. Memory values are read from
configs/hbm2e_pfm.yaml, and every bit count is calculated from a size.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hbm2e_pfm.yaml"

# Choices that are not part of the Ramulator config.
PAGE_BYTES = 2 * 1024 * 1024  # 2 MB superpage (paper page 6)
TILE_N = 4096                 # one tile, layout is column major (paper page 9)
ELEMENT_BYTES = 1             # INT8

# Fixed by the memory type, so Ramulator's config does not list them.
HBM2_PREFETCH = 2
DDR_BEATS_PER_CLOCK = 2

BITS_PER_BYTE = 8
MT_PER_GT = 1000


def bits_for(count):
    """Return the address bits needed to select one of count places.

    Raises ValueError unless count is a power of two, since no whole number
    of bits can select between, for example, 12 places.
    """
    if count <= 0 or count & (count - 1):
        raise ValueError(f"{count} is not a power of two")
    return count.bit_length() - 1


with open(CONFIG_PATH) as f:
    DRAM = yaml.safe_load(f)["MemorySystem"]["DRAM"]
ORG = DRAM["org"]
TIMING = DRAM["timing"]

REQUEST_BYTES = HBM2_PREFETCH * ORG["channel_width"] // BITS_PER_BYTE
REQUESTS_PER_ROW = ORG["column"] // HBM2_PREFETCH

PAGE_BITS = bits_for(PAGE_BYTES)
OFFSET_BITS = bits_for(REQUEST_BYTES)
ASSIGNABLE_BITS = range(OFFSET_BITS, PAGE_BITS)

# Most address bits each memory level can use (paper page 6).
LEVEL_CAPACITY = {
    "col": bits_for(REQUESTS_PER_ROW),
    "row": bits_for(ORG["row"]),
    "bk": bits_for(ORG["bank"]),
    "bg": bits_for(ORG["bankgroup"]),
    "pch": bits_for(ORG["pseudochannel"]),
    "ch": bits_for(ORG["channel"]),
}

# Widths when the page uses every bank and channel. Row gets the bits left over.
FULL_BANK_WIDTHS = dict(LEVEL_CAPACITY)
FULL_BANK_WIDTHS["row"] = 0
FULL_BANK_WIDTHS["row"] = len(ASSIGNABLE_BITS) - sum(FULL_BANK_WIDTHS.values())
if not 0 <= FULL_BANK_WIDTHS["row"] <= LEVEL_CAPACITY["row"]:
    raise ValueError(f"{PAGE_BYTES} B page does not fit the config's levels")

TILE_K = PAGE_BYTES // (TILE_N * ELEMENT_BYTES)

# Address bits. Page 6 gives p_n = log2(N); we include the element size
# because addresses count bytes. Both give the same result for INT8.
P_N = bits_for(TILE_N * ELEMENT_BYTES)   # contiguous address bits
P_K = PAGE_BITS - P_N    # stride address bits

# Timings (in ns) used by the NPU score (paper Table I).
T_RCD = TIMING["tRCDRD"]
T_CCD_S = TIMING["tCCDS"]
T_CCD_L = TIMING["tCCDL"]


def peak_gbps():
    """Return peak bandwidth in GB/s, computed from the Ramulator config.

    Paper page 7: B_peak = (#channels * #pins per channel) * frequency.

    The config has no pins field, because channel_width only sets the request
    size in Ramulator v2.0a. Pins come from how a request crosses the bus:
    pins = request bits / beats, where beats = beats per clock * nBL.

    The paper treats pseudo-channels as channels.
    """
    channels = ORG["channel"] * ORG["pseudochannel"]
    beats = DDR_BEATS_PER_CLOCK * TIMING["nBL"]
    pins = REQUEST_BYTES * BITS_PER_BYTE / beats
    gbps_per_pin = TIMING["rate"] / MT_PER_GT
    return channels * pins * gbps_per_pin / BITS_PER_BYTE


PEAK_GBPS = peak_gbps()


def build_mapping(order, widths=FULL_BANK_WIDTHS):
    """Return a mapping that gives each level its bits, in order, low to high."""
    mapping = {}
    next_bit = OFFSET_BITS
    for level in order:
        mapping[level] = list(range(next_bit, next_bit + widths[level]))
        next_bit += widths[level]
    return mapping


# Fig. 4(a), low bits to high bits. The order inside channel (ch, pch) and
# bank (bg, bk) is our choice, because Fig. 4 shows them as one level each.
NPU_FRIENDLY = build_mapping(("ch", "pch", "bg", "bk", "col", "row"))
PIM_FRIENDLY = build_mapping(("col", "ch", "pch", "bg", "bk", "row"))


def check_mapping(mapping):
    """
        Raise ValueError unless mapping is valid.

        mapping is a dict of level name -> list of address bits.
        Each bit must go to one level, and each level can use at most
        its capacity (paper page 6). A level can be left with no bits.
    """
    if set(mapping) != set(LEVEL_CAPACITY):
        raise ValueError(f"levels must be {sorted(LEVEL_CAPACITY)}")

    for level, bits in mapping.items():
        if len(bits) > LEVEL_CAPACITY[level]:
            raise ValueError(f"{level} has {len(bits)} bits, at most {LEVEL_CAPACITY[level]} allowed")

    used = []
    for bits in mapping.values():
        used.extend(bits)
    used.sort()

    if used != list(ASSIGNABLE_BITS):
        raise ValueError(f"bits must be {OFFSET_BITS}-{PAGE_BITS - 1}, each used once")


if __name__ == "__main__":
    print(f"capacity {LEVEL_CAPACITY}, {len(ASSIGNABLE_BITS)} bits to assign")
    print(f"full bank widths {FULL_BANK_WIDTHS}")
    print(f"tile {TILE_N} x {TILE_K}: {TILE_N * TILE_K * ELEMENT_BYTES} B, p_n = {P_N}, p_k = {P_K}")
    print(f"peak {PEAK_GBPS} GB/s")

    check_mapping(PIM_FRIENDLY)
    print("valid mapping accepted")

    # One bit too many for ch, to test the check.
    too_many = {**PIM_FRIENDLY, "ch": PIM_FRIENDLY["col"][-1:] + PIM_FRIENDLY["ch"]}
    try:
        check_mapping(too_many)
    except ValueError as e:
        print(f"Bad mapping rejected: {e}")