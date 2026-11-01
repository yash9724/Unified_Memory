"""PIM score of one address mapping (paper pages 7-8, eq. 5-9).

The score says how well a mapping suits PIM, where each bank computes on its
own data. It multiplies two fractions: how many banks hold useful data, and
how close a bank's time is to a reference time.

Equations, from the top down:

  eq. 5  PIM(f) = (#active banks / #banks total) * (T_ref / T_bank)

  eq. 6  #active banks = 2^(bits in levels at or above bank)
         Those levels are bk, bg, pch and ch. With fewer of these bits, the
         data sits in fewer banks and the other banks have nothing to compute.

  eq. 7  T_ref = (2^(p_n + p_k) / (#active banks * N_row)) * tRCD + N_col * tCCD
         N_row is the number of rows in each bank and N_col the number of
         columns in each row. With 512 active banks and 32768 rows this gives
         0.125 row opens, so T_ref is below any real bank time and the second
         fraction in eq. 5 stays below 1.

  eq. 8  T_bank = max(T_bankaccess, T_io)
         The slower of the two stages limits the bank.
         T_bankaccess = 2^(row bits) * tRCD + 2^(col bits) * tCCD

  eq. 9  T_io = 2^p_n / (2^(channel bits below p_n) * B_ch)
              + 2^p_k / (2^(channel bits at or above p_n) * B_ch)
         Time to move data through the channels. It grows when few channel
         bits fall in that part of the address, as less of the data is spread
         across channels. B_ch is the bandwidth of one channel.

The paper treats pseudo-channels as channels, so channel bits include pch and
B_ch is peak bandwidth divided by the number of pseudo-channels. tCCD is
tCCD_L because reads inside one bank never change bank group.

Worked example, PIM-friendly layout with the current config:

  #active banks = 2^(2 + 2 + 1 + 4) = 512, #banks total = 512
  T_ref = (2097152 / (512 * 32768)) * 14 + 32 * 3.5 = 113.75 ns
  T_bankaccess = 4 * 14 + 32 * 3.5 = 168 ns
  T_io = 4096 / (4 * 12.8) + 512 / (8 * 12.8) = 85 ns
  score = (512 / 512) * (113.75 / 168) = 0.677
"""

from mapper.model import (
    FULL_BANK_WIDTHS,
    NPU_FRIENDLY,
    ORG,
    P_K,
    P_N,
    PEAK_GBPS,
    PIM_FRIENDLY,
    REQUESTS_PER_ROW,
    T_CCD_L,
    T_RCD,
    build_mapping,
    check_mapping,
)

CHANNELS = ORG["channel"] * ORG["pseudochannel"]
TOTAL_BANKS = CHANNELS * ORG["bankgroup"] * ORG["bank"]
B_CH = PEAK_GBPS / CHANNELS  # GB/s, which equals bytes per ns


def active_banks(mapping):
    """Return #active banks for a mapping (eq. 6)."""
    bits = 0
    for level in ("bk", "bg", "pch", "ch"):
        bits += len(mapping[level])
    return 2 ** bits


def t_ref(mapping):
    """Return the reference time of one bank in ns (eq. 7)."""
    row_opens = 2 ** (P_N + P_K) / (active_banks(mapping) * ORG["row"])
    return row_opens * T_RCD + REQUESTS_PER_ROW * T_CCD_L


def t_bankaccess(mapping):
    """Return the time to open rows and read columns in one bank in ns."""
    return 2 ** len(mapping["row"]) * T_RCD + 2 ** len(mapping["col"]) * T_CCD_L


def t_io(mapping):
    """Return the time to move data through the channels in ns (eq. 9)."""
    channels_low = 0
    channels_high = 0
    for bit in mapping["ch"] + mapping["pch"]:
        if bit < P_N:
            channels_low += 1
        else:
            channels_high += 1

    return (2 ** P_N / (2 ** channels_low * B_CH)
            + 2 ** P_K / (2 ** channels_high * B_CH))


def pim_score(mapping):
    """Return PIM(f) for a mapping (eq. 5, 8)."""
    t_bank = max(t_bankaccess(mapping), t_io(mapping))
    return active_banks(mapping) / TOTAL_BANKS * t_ref(mapping) / t_bank


if __name__ == "__main__":
    # Test layouts only, not optimizer results.
    # All channel bits high, so little data is spread across channels.
    low_spread = build_mapping(("col", "row", "bg", "bk", "ch", "pch"))
    # The bk bits move to row, so a quarter of the banks hold data.
    fewer_banks = build_mapping(
        ("col", "ch", "pch", "bg", "bk", "row"),
        {**FULL_BANK_WIDTHS, "bk": 0, "row": FULL_BANK_WIDTHS["row"] + FULL_BANK_WIDTHS["bk"]},
    )

    layouts = {
        "NPU-friendly": NPU_FRIENDLY,
        "PIM-friendly": PIM_FRIENDLY,
        "low channel spread": low_spread,
        "fewer banks": fewer_banks,
    }
    for name, mapping in layouts.items():
        check_mapping(mapping)
        print(f"{name}: active banks {active_banks(mapping)}, T_ref {t_ref(mapping)}, "
              f"T_bankaccess {t_bankaccess(mapping)}, T_io {t_io(mapping)}, "
              f"score {pim_score(mapping):.3f}")