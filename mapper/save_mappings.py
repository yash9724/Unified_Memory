"""Save the three weight-matrix layouts to mapper/out/mappings.json.

Later steps (the Ramulator plugin, the traces and the final runs) read this
file, so they all use the same mappings.

  NPU-friendly and PIM-friendly come from Fig. 4(a) of the paper, built in
  model.py. The solver cannot produce them both under the current scores.

  The jointly optimized layout comes from the solver (eq. 10).

The paper picks alpha by profiling operators and gives no value we can use, so
JOINT_ALPHA is our choice. The sweep prints the scores at each alpha to show
how much the choice matters.
"""

import json
from pathlib import Path

from mapper.model import NPU_FRIENDLY, PIM_FRIENDLY, check_mapping
from mapper.npu_score import npu_score, whole_page_counts
from mapper.pim_score import pim_score
from mapper.solver import solve

OUT_PATH = Path(__file__).resolve().parent / "out" / "mappings.json"

# Alpha values to try. Zero is left out because nothing in the objective would
# then reward PIM, so the solver could return a layout that leaves banks unused.
ALPHAS = (0.5, 1, 2, 5, 10, 100)
JOINT_ALPHA = 1

# Largest difference allowed when comparing two scores, as they are decimals.
TOLERANCE = 1e-9


def describe(mapping):
    """Return what gets saved for one layout: its bits and both of its scores."""
    return {
        "bits": mapping,
        "npu_score": npu_score(mapping, whole_page_counts),
        "pim_score": pim_score(mapping),
    }


def sweep():
    """Run the solver at each alpha and print the scores of what it returns.

    Returns the mapping found at JOINT_ALPHA, which is the joint layout.
    """
    joint = None
    for alpha in ALPHAS:
        # solve also returns the solver's own value, which is not needed here
        # because step 9 already checked it against the score functions.
        mapping, _ = solve(alpha)
        check_mapping(mapping)
        result = describe(mapping)
        print(f"alpha {alpha}: NPU {result['npu_score']:.3f}, PIM {result['pim_score']:.3f}")
        if alpha == JOINT_ALPHA:
            joint = mapping
    return joint


def check_saved(path):
    """Read the file back and raise ValueError unless every saved score is right.

    This catches a mapping that changed on its way to the file, since a layout
    used by later steps must score the same as the one chosen here.
    """
    with open(path) as f:
        saved = json.load(f)

    for name, layout in saved["layouts"].items():
        check_mapping(layout["bits"])
        fresh = describe(layout["bits"])
        for score in ("npu_score", "pim_score"):
            if abs(layout[score] - fresh[score]) > TOLERANCE:
                raise ValueError(f"{name}: saved {score} does not match the score functions")
    return saved


if __name__ == "__main__":
    joint = sweep()

    # The two baselines must also be valid before they are saved.
    check_mapping(NPU_FRIENDLY)
    check_mapping(PIM_FRIENDLY)

    output = {
        "joint_alpha": JOINT_ALPHA,
        "layouts": {
            "npu_friendly": describe(NPU_FRIENDLY),
            "pim_friendly": describe(PIM_FRIENDLY),
            "joint": describe(joint),
        },
    }

    # Create mapper/out on the first run, and keep the file readable by eye.
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    saved = check_saved(OUT_PATH)
    print(f"saved to {OUT_PATH}")
    for name, layout in saved["layouts"].items():
        print(f"{name}: NPU {layout['npu_score']:.3f}, PIM {layout['pim_score']:.3f}, bits {layout['bits']}")
    print("checks passed")