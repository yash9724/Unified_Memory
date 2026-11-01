"""NPU score of one address mapping (paper page 7, eq. 1-4).

The score says how fast the NPU can read one 2 MB page under a mapping,
compared with the peak bandwidth of the memory. The solver uses it to
compare mappings, so what matters most here is the order of the scores.

Equations, from the top down:

  eq. 1  NPU(f) = B_NPU / B_peak
         Fraction of peak bandwidth the mapping achieves.

         B_NPU = data size / T_mem(f)
         Bandwidth is bytes moved divided by the time taken.

  eq. 2  T_mem(f) = max over pseudo-channels of T_access(pch, f)
         Pseudo-channels work at the same time, so the page is finished
         when the slowest one is finished. Every pseudo-channel gets the
         same number of banks here, so they all take the same time.

         T_access(pch, f) = banks per pseudo-channel * T_access(bank, f)
         Page 7 says banks inside a pseudo-channel operate serially and
         that T_access(pch) aggregates over them, but prints no formula.
         Adding them up is our reading of that text.

  eq. 3  T_access(bank, f) = #row-act * tRCD + #col * tCCD
         A bank spends its time opening rows and reading columns.

  eq. 4  #row-act = 2^(row bits below p_n) * 2^(col bits at or above p_n)
         #col     = 2^(col bits)
         p_n splits the address into the contiguous part (low bits) and
         the stride part (high bits). For our tile, p_n = log2(4096 * 1 B) = 12.
         A row bit in the contiguous part, or a column bit in the stride
         part, adds row changes while the tile is read.

Worked example, NPU-friendly layout with whole-page counts, for the current
config (peak_gbps in model.py computes the 409.6 GB/s from the config file):

  row opens = 2^2 = 4, reads = 2^(5 + 2) = 128, tCCD = 1.25 ns
  T_bank = 4 * 14 + 128 * 1.25 = 216 ns
  T_pch  = 16 * 216 = 3456 ns
  score  = (2097152 B / 3456 ns) / 409.6 GB/s = 1.481

A score above 1 is not real bandwidth. tCCD_S = 1.25 ns is used as printed
in Table I, but one 32 B burst takes 2.5 ns on the bus, and eq. 3 has no term
for that limit. Ramulator measures the real bandwidth later.
"""

from mapper.model import (
    ASSIGNABLE_BITS,
    NPU_FRIENDLY,
    P_N,
    PAGE_BYTES,
    PEAK_GBPS,
    PIM_FRIENDLY,
    T_CCD_L,
    T_CCD_S,
    T_RCD,
    check_mapping,
)


def paper_counts(mapping):
    """Return (row opens, reads) per bank, exactly as eq. 4 and #col are printed.

    Example, NPU-friendly layout:
      row bits 19, 20 are not below 12, so row_low = 0
      col bits 14-18 are all at or above 12, so col_high = 5
      row opens = 2^0 * 2^5 = 32, reads = 2^5 = 32

    Example, PIM-friendly layout:
      row_low = 0, and col bits 5-9 are all below 12, so col_high = 0
      row opens = 2^0 * 2^0 = 1, reads = 2^5 = 32

    Reads stay at 32 for any mapping with 5 column bits, as both layouts have.
    A bank holding 4 full rows really serves 128 reads, so these counts do not
    cover the whole page. That makes the PIM-friendly score too high.
    """
    # Row bits in the contiguous part of the address.
    row_low = 0
    for bit in mapping["row"]:
        if bit < P_N:
            row_low += 1

    # Column bits in the stride part of the address.
    col_high = 0
    for bit in mapping["col"]:
        if bit >= P_N:
            col_high += 1

    row_opens = 2 ** row_low * 2 ** col_high
    reads = 2 ** len(mapping["col"])
    return row_opens, reads


def whole_page_counts(mapping):
    """Return (row opens, reads) per bank for reading the whole 2 MB page.

    n bits select one of 2^n places, so:
      2 row bits give 2^2 = 4 rows per bank, each opened once
      5 col + 2 row bits give 2^7 = 128 reads per bank

    Check: 32 pseudo-channels * 16 banks * 128 reads * 32 B = 2 MB.

    Not printed in the paper. It is our reading of the formulas, supported by
    page 10, which gives four row activations per bank for the whole page.
    This assumes each row is opened only once.
    """
    row_opens = 2 ** len(mapping["row"])
    reads = 2 ** (len(mapping["col"]) + len(mapping["row"]))
    return row_opens, reads


def pick_tccd(mapping):
    """Return the gap between consecutive reads: tCCD_S or tCCD_L, in ns.

    Page 7: tCCD_S applies across different bank groups and tCCD_L within the
    same bank group. It gives no rule in terms of address bits, so this rule
    is ours, not the paper's.

    A pseudo-channel sees only its own reads, so ch and pch bits are skipped.
    The lowest remaining bit changes on every read in that pseudo-channel:
      it is a bg bit -> every read changes bank group -> tCCD_S
      any other bit  -> reads stay in one bank group  -> tCCD_L

    Example: NPU-friendly, bit 10 is a bg bit -> 1.25 ns.
    Example: PIM-friendly, bit 5 is a col bit -> 3.5 ns.
    """
    channel_bits = mapping["ch"] + mapping["pch"]
    for bit in ASSIGNABLE_BITS:
        if bit not in channel_bits:
            if bit in mapping["bg"]:
                return T_CCD_S
            return T_CCD_L
    raise ValueError("mapping has only channel bits")


def npu_score(mapping, counts):
    """Return NPU(f) for a mapping (eq. 1-3).

    counts is the function that supplies the row opens and reads, either
    paper_counts or whole_page_counts.
    """
    row_opens, reads = counts(mapping)
    tccd = pick_tccd(mapping)

    # Eq. 3: time for one bank, in ns.
    t_bank = row_opens * T_RCD + reads * tccd

    # Time for one pseudo-channel: its banks work one after another.
    # 2 bg bits + 2 bk bits give 2^4 = 16 banks.
    banks = 2 ** (len(mapping["bg"]) + len(mapping["bk"]))
    t_pch = banks * t_bank

    # Eq. 2: all pseudo-channels take the same time, so T_mem = t_pch.
    # Eq. 1: bandwidth, then the fraction of peak. Bytes per ns equals GB/s.
    bandwidth = PAGE_BYTES / t_pch

    return bandwidth / PEAK_GBPS


if __name__ == "__main__":
    layouts = {"NPU": NPU_FRIENDLY, "PIM": PIM_FRIENDLY}
    versions = {"paper formula": paper_counts, "whole page": whole_page_counts}

    # Stop here if a layout is invalid, so it never gets a score.
    for mapping in layouts.values():
        check_mapping(mapping)

    for version, counts in versions.items():
        for name, mapping in layouts.items():
            row_opens, reads = counts(mapping)
            score = npu_score(mapping, counts)
            print(f"{name} {version}: row opens {row_opens}, reads {reads}, "
                  f"tCCD {pick_tccd(mapping)}, score {score:.3f}")