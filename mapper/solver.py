"""Address mapping optimizer (paper page 8, eq. 10).

Finds the mapping with the highest NPU(f) + alpha * PIM(f).

  X[level][bit] = 1 when an address bit goes to a memory level. These are the
  yes/no choices the paper solves for (page 6), with two rules:
    each bit goes to exactly one level
    each level uses at most its capacity

PuLP only handles linear problems, and both scores use powers of 2, division
and max. The paper does not say how it deals with this, so the method below
is ours.

The scores do not depend on exactly which bits a level holds. They depend on
a few counts:

  col            column bits
  bank           bg + bk bits
  channel        ch + pch bits
  channel_low    channel bits below p_n
  uses_tccd_s    whether the lowest non-channel bit is a bg bit

Row bits are whatever is left. There are few enough combinations of these
counts to score each one in advance, using npu_score and pim_score. The solver
gets one yes/no variable per combination and picks exactly one. A combination's
score is then a fixed number, so the objective is linear.

The rules below tie the picked combination to the X variables, so the bits
must really produce the counts of that combination.

Example: picking col = 5, bank = 4, channel = 5, channel_low = 5 and
uses_tccd_s = yes forces 5 col bits, 4 bg + bk bits and 5 channel bits, all 5
channel bits below p_n, with a bg bit as the lowest non-channel bit.
NPU-friendly is one mapping that meets all of this.
"""

from collections import namedtuple

import pulp

from mapper.model import (
    ASSIGNABLE_BITS,
    LEVEL_CAPACITY,
    P_N,
    T_CCD_S,
    check_mapping,
)
from mapper.npu_score import npu_score, pick_tccd, whole_page_counts
from mapper.pim_score import pim_score

Counts = namedtuple("Counts", "col bank channel channel_low uses_tccd_s")

LOW_BITS = [bit for bit in ASSIGNABLE_BITS if bit < P_N]
HIGH_BITS = [bit for bit in ASSIGNABLE_BITS if bit >= P_N]
CHANNEL_LEVELS = ("ch", "pch")
BANK_LEVELS = ("bg", "bk")


def counts_of(mapping):
    """Return the counts of a mapping, which fix both of its scores."""
    channel_bits = mapping["ch"] + mapping["pch"]
    return Counts(
        col=len(mapping["col"]),
        bank=len(mapping["bg"]) + len(mapping["bk"]),
        channel=len(channel_bits),
        channel_low=sum(bit < P_N for bit in channel_bits),
        uses_tccd_s=pick_tccd(mapping) == T_CCD_S,
    )


def example_mapping(counts):
    """Return one mapping with these counts, or None if no mapping has them.

    Used only to score a combination, so which bits go where does not matter.
    """
    channel_high = counts.channel - counts.channel_low
    row = len(ASSIGNABLE_BITS) - counts.col - counts.bank - counts.channel
    if channel_high > len(HIGH_BITS) or counts.channel_low > len(LOW_BITS):
        return None
    if not 0 <= row <= LEVEL_CAPACITY["row"]:
        return None

    # Channel bits first, because the counts say where they must sit.
    channel_bits = LOW_BITS[:counts.channel_low] + HIGH_BITS[:channel_high]
    pch = min(counts.channel, LEVEL_CAPACITY["pch"])
    mapping = {"pch": channel_bits[:pch], "ch": channel_bits[pch:]}

    # The rest go low to high. A bg bit first gives tCCD_S, and any other first
    # gives tCCD_L.
    bg = min(counts.bank, LEVEL_CAPACITY["bg"])
    widths = {"bg": bg, "bk": counts.bank - bg, "col": counts.col, "row": row}
    order = ("bg", "bk", "col", "row") if counts.uses_tccd_s else ("row", "col", "bk", "bg")
    left = [bit for bit in ASSIGNABLE_BITS if bit not in channel_bits]
    for level in order:
        mapping[level] = left[:widths[level]]
        left = left[widths[level]:]

    if counts_of(mapping) != counts:
        return None
    check_mapping(mapping)
    return mapping


def all_combinations():
    """Return every combination of counts that some mapping can have."""
    combinations = []
    bank_capacity = sum(LEVEL_CAPACITY[level] for level in BANK_LEVELS)
    channel_capacity = sum(LEVEL_CAPACITY[level] for level in CHANNEL_LEVELS)
    for col in range(LEVEL_CAPACITY["col"] + 1):
        for bank in range(bank_capacity + 1):
            for channel in range(channel_capacity + 1):
                for channel_low in range(channel + 1):
                    for uses_tccd_s in (False, True):
                        # No bank bits means no bg bit that could come first.
                        if uses_tccd_s and bank == 0:
                            continue
                        counts = Counts(col, bank, channel, channel_low, uses_tccd_s)
                        if example_mapping(counts) is not None:
                            combinations.append(counts)
    return combinations


def objective(mapping, alpha):
    """Return NPU(f) + alpha * PIM(f) for a mapping (eq. 10)."""
    return npu_score(mapping, whole_page_counts) + alpha * pim_score(mapping)


def solve(alpha):
    """Return the mapping with the highest objective, and that objective."""
    problem = pulp.LpProblem("address_mapping", pulp.LpMaximize)

    x = {(level, bit): problem.add_variable(f"x_{level}_{bit}", cat="Binary")
         for level in LEVEL_CAPACITY for bit in ASSIGNABLE_BITS}

    # Paper page 6: each bit goes to exactly one level.
    for bit in ASSIGNABLE_BITS:
        problem += pulp.lpSum(x[level, bit] for level in LEVEL_CAPACITY) == 1

    # Paper page 6: each level uses at most its capacity.
    for level, capacity in LEVEL_CAPACITY.items():
        problem += pulp.lpSum(x[level, bit] for bit in ASSIGNABLE_BITS) <= capacity

    # first[bit] = 1 when bit is the lowest non-channel bit.
    first = {bit: problem.add_variable(f"first_{bit}", cat="Binary") for bit in ASSIGNABLE_BITS}
    uses_tccd_s = problem.add_variable("uses_tccd_s", cat="Binary")
    problem += pulp.lpSum(first.values()) == 1
    for bit in ASSIGNABLE_BITS:
        # The first bit is not a channel bit...
        problem += first[bit] + x["ch", bit] + x["pch", bit] <= 1
        # ...and every bit below it is.
        for lower in ASSIGNABLE_BITS:
            if lower < bit:
                problem += first[bit] <= x["ch", lower] + x["pch", lower]
        # When first[bit] = 1 these two reduce to uses_tccd_s = x["bg", bit].
        # When first[bit] = 0 they allow anything.
        problem += uses_tccd_s >= first[bit] + x["bg", bit] - 1
        problem += uses_tccd_s <= 1 - first[bit] + x["bg", bit]

    # One yes/no variable per combination, and exactly one is picked.
    combinations = all_combinations()
    pick = {counts: problem.add_variable(f"pick_{i}", cat="Binary")
            for i, counts in enumerate(combinations)}
    problem += pulp.lpSum(pick.values()) == 1

    def picked(field):
        """Return the value of a count in the picked combination."""
        return pulp.lpSum(getattr(counts, field) * pick[counts] for counts in combinations)

    # The bits must produce the counts of the picked combination.
    problem += pulp.lpSum(x["col", bit] for bit in ASSIGNABLE_BITS) == picked("col")
    problem += pulp.lpSum(x[level, bit] for level in BANK_LEVELS
                          for bit in ASSIGNABLE_BITS) == picked("bank")
    problem += pulp.lpSum(x[level, bit] for level in CHANNEL_LEVELS
                          for bit in ASSIGNABLE_BITS) == picked("channel")
    problem += pulp.lpSum(x[level, bit] for level in CHANNEL_LEVELS
                          for bit in LOW_BITS) == picked("channel_low")
    problem += uses_tccd_s == picked("uses_tccd_s")

    # Eq. 10, with each combination's score worked out in advance.
    problem += pulp.lpSum(objective(example_mapping(counts), alpha) * pick[counts]
                          for counts in combinations)

    # CBC says "optimal within gap tolerance" when it proves the best answer,
    # which PuLP reports as GapLimit. The checks in __main__ confirm the result.
    result = problem.solve(pulp.COIN_CMD(msg=False))
    if result.status not in (pulp.LpSolveStatus.Optimal, pulp.LpSolveStatus.GapLimit):
        raise RuntimeError(f"solver stopped with status {result.status}")

    mapping = {level: [bit for bit in ASSIGNABLE_BITS if x[level, bit].value() > 0.5]
               for level in LEVEL_CAPACITY}
    return mapping, problem.objective.value()


if __name__ == "__main__":
    # alpha = 1 only tests the solver. It is not from the paper, which picks
    # alpha by profiling operators.
    ALPHA = 1
    TOLERANCE = 1e-6

    mapping, solver_value = solve(ALPHA)
    check_mapping(mapping)
    print(f"mapping {mapping}")
    print(f"counts {counts_of(mapping)}")

    # The solver's value must match the score functions on the mapping it returned.
    npu = npu_score(mapping, whole_page_counts)
    pim = pim_score(mapping)
    value = objective(mapping, ALPHA)
    print(f"NPU {npu:.3f}, PIM {pim:.3f}, objective {value:.3f}, solver said {solver_value:.3f}")
    if abs(value - solver_value) > TOLERANCE:
        raise ValueError("solver value does not match the score functions")

    # No other combination may score higher.
    combinations = all_combinations()
    best = max(objective(example_mapping(counts), ALPHA) for counts in combinations)
    print(f"{len(combinations)} combinations, best objective {best:.3f}")
    if value < best - TOLERANCE:
        raise ValueError("another combination scores higher than the solver's mapping")
    print("checks passed")