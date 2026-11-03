"""Make the result figures and tables from results/results.csv.

Two sets, each as a PDF figure (for LaTeX), a 300 dpi PNG (for slides) and a
LaTeX table:

  results/baseline.*      the two Fig. 4(a) layouts: NPU-friendly, PIM-friendly
  results/contribution.*  the same two plus the solver's joint layout

Run after scripts/run_all.py:  .venv/bin/python -m scripts.plot_results
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "results.csv"

# Fixed label and colour per layout, so a layout looks the same in every figure.
LAYOUTS = {
    "npu_friendly": ("NPU-friendly", "#2a78d6"),
    "pim_friendly": ("PIM-friendly", "#eb6834"),
    "joint": ("Joint (PFM)", "#1baf7a"),
}
FIGURES = {
    "baseline": ["npu_friendly", "pim_friendly"],
    "contribution": ["npu_friendly", "pim_friendly", "joint"],
}
DEVICES = {"NPU": "NPU sequential read", "PIM": "PIM GEMV (all-bank MAC)"}
TEXT = "#3d3d3a"
GRID = "#e4e3dd"


def read_results():
    """Return (shapes in file order, {(shape, layout, device): utilization in %})."""
    shapes = []
    values = {}
    with open(RESULTS_PATH) as f:
        for row in csv.DictReader(f):
            if row["shape"] not in shapes:
                shapes.append(row["shape"])
            values[(row["shape"], row["layout"], row["device"])] = float(row["utilization"]) * 100
    return shapes, values


def shape_label(shape):
    """Turn "14336x4096" into two lines, so long shapes do not run into each other."""
    return shape.replace("x", "\n×")


def plot(shapes, values, layouts, name):
    """Draw NPU and PIM side by side, one group of bars per shape."""
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": TEXT,
                         "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    width = 0.26

    for ax, (device, title) in zip(axes, DEVICES.items()):
        for i, layout in enumerate(layouts):
            label, colour = LAYOUTS[layout]
            # Centre the group of bars on the tick.
            shift = (i - (len(layouts) - 1) / 2) * (width + 0.02)
            xs = [s + shift for s in range(len(shapes))]
            ys = [values[(shape, layout, device)] for shape in shapes]
            bars = ax.bar(xs, ys, width, color=colour, label=label, zorder=2)
            # Values on the bars, so the colours never carry the numbers alone.
            ax.bar_label(bars, fmt="%.1f", padding=1.5, fontsize=5.5, color=TEXT)

        ax.set_title(title, fontsize=8.5, color=TEXT)
        ax.set_xticks(range(len(shapes)), [shape_label(s) for s in shapes])
        ax.set_xlabel("Weight matrix (rows × cols, INT8)")
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Bandwidth utilization (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(layouts), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    fig.savefig(RESULTS_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table(shapes, values, layouts, name):
    """Write a LaTeX table: one row per shape, NPU and PIM columns per layout."""
    lines = [
        r"\begin{tabular}{l" + "cc" * len(layouts) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{LAYOUTS[layout][0]}}}" for layout in layouts) + r" \\",
        "Shape" + " & NPU & PIM" * len(layouts) + r" \\",
        r"\midrule",
    ]
    for shape in shapes:
        cells = [f"{values[(shape, layout, device)]:.1f}" for layout in layouts for device in DEVICES]
        lines.append(shape.replace("x", r"$\times$") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / f"{name}_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    shapes, values = read_results()
    for name, layouts in FIGURES.items():
        plot(shapes, values, layouts, name)
        write_table(shapes, values, layouts, name)
    print(f"wrote figures and tables to {RESULTS_DIR}")
