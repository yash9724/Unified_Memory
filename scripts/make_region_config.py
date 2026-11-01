"""Write a Ramulator config that gives each saved layout its own memory region.

Reads configs/hbm2e_pfm.yaml and mapper/out/mappings.json, and writes
configs/hbm2e_pfm_regions.yaml. Only the AddrMapper section differs, so memory
values stay in one place. Run this again whenever either input changes.

The region size is our choice, not the paper's. Each region must hold the
largest matrix, 4096 x 14336 at 1 byte per element, which is about 56 MB.
"""

import json
from pathlib import Path

import yaml

from mapper.model import CONFIG_PATH, PAGE_BYTES

ROOT = Path(__file__).resolve().parent.parent
MAPPINGS_PATH = ROOT / "mapper" / "out" / "mappings.json"
OUT_PATH = ROOT / "configs" / "hbm2e_pfm_regions.yaml"

REGION_BYTES = 1024 * 1024 * 1024  # 1 GB

# Our short level names -> the names Ramulator's HBM2 defines.
RAMULATOR_LEVELS = {
    "ch": "channel",
    "pch": "pseudochannel",
    "bg": "bankgroup",
    "bk": "bank",
    "row": "row",
    "col": "column",
}


def make_regions(layouts):
    """Return one region per layout, placed one after another from address 0."""
    regions = []
    for index, (name, layout) in enumerate(layouts.items()):
        regions.append({
            "name": name,
            "base": index * REGION_BYTES,
            "size": REGION_BYTES,
            "bits": {RAMULATOR_LEVELS[level]: bits for level, bits in layout["bits"].items()},
        })
    return regions


if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(MAPPINGS_PATH) as f:
        layouts = json.load(f)["layouts"]

    config["MemorySystem"]["AddrMapper"] = {
        "impl": "RegionMapper",
        "page_bytes": PAGE_BYTES,
        "regions": make_regions(layouts),
    }

    # sort_keys=False keeps the sections in their original order, and
    # default_flow_style=None writes each list of bits on one line.
    with open(OUT_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)

    for region in config["MemorySystem"]["AddrMapper"]["regions"]:
        print(f"{region['name']}: base {region['base']:#x}, size {region['size']:#x}")
    print(f"wrote {OUT_PATH}")