"""
Shared definitions for address mapping optimizer.

Levels follow Fig. 9 of the paper. Their widths match configs/hbm2e_pfm.yaml
"""

from math import log2

PAGE_BITS = 21     # 2 MB superpage
OFFSET_BITS = 5    # 32B burst
ASSIGNABLE_BITS = range(OFFSET_BITS, PAGE_BITS)

# Address bits needed by each memory level.
LEVEL_WIDTHS = {"col" : 5, "row": 2, "bk": 2, "bg": 2, "pch": 1, "ch": 4}

# one tile, layout is column major (paper page 9)
TILE_N = 4096
TILE_K = 512
ELEMENT_BYTES = 1     # INT8

# address bits (paper page 6)
P_N = int(log2(TILE_N))  # contigous address bits = 2^12 = 4096 = TILE_N
P_K = PAGE_BITS - P_N    # stride address bits = 2^9 = 512 = TILE_K

# Timings(in ns) used by the NPU score (paper Table I).
T_RCD = 14
T_CCD_S = 1.25
T_CCD_L = 3.5

def check_mapping(mapping):
    """
        Raise ValueError unless mapping is valid.

        mapping is a dict of level name -> list of address bits.
        Each bit must go to one level. The paper requires each level
        to use at most its width; since the widths add up to the 
        number of bits, each level uses exactly its width.
    """
    if set(mapping) != set(LEVEL_WIDTHS):
        raise ValueError(f"levels must be {sorted(LEVEL_WIDTHS)}")

    for level, bits in mapping.items():
        if len(bits) != LEVEL_WIDTHS[level]:
            raise ValueError(f"{level} has {len(bits)} bits, needs {LEVEL_WIDTHS[level]}")

    used = []
    for bits in mapping.values():
        used.extend(bits)
    used.sort()

    if used != list(ASSIGNABLE_BITS):
        raise ValueError(f"bits must be {OFFSET_BITS}-{PAGE_BITS - 1}, each used once")


if __name__ == "__main__":
    widths = " + ".join(str(w) for w in LEVEL_WIDTHS.values())
    print(f"{widths} = {sum(LEVEL_WIDTHS.values())}")
    print(f"tile {TILE_N} x {TILE_K}: {TILE_N * TILE_K * ELEMENT_BYTES} B, p_n = {P_N}, p_k = {P_K}")

    # Only an example for testing the check, not an optimizer result.
    valid = {
    "col": [5, 6, 7, 8, 9],
    "ch": [10, 11, 12, 13],
    "pch": [14],
    "bg": [15, 16],
    "bk": [17, 18],
    "row": [19, 20],
    }
    check_mapping(valid)
    print("valid mapping accepted")

    too_many = {**valid, "ch": [9, 10, 11, 12, 13]}
    try:
        check_mapping(too_many)
    except ValueError as e:
        print(f"Bad mapping rejected: {e}")    