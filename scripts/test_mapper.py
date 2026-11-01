"""Check RegionMapper's decoding against values worked out by hand.

A wrong decode would not crash. It would send reads to the wrong places and
spoil every result, so this test runs real addresses through Ramulator.

The expected values below are written out on purpose, not calculated from the
layouts, because calculating them with the same logic would test nothing. They
match the layouts in mapper/out/mappings.json. If those layouts change, this
table must be worked out again.

Layouts, as level: address bits, low to high:

  npu_friendly (base 0x0)         ch 5-8, pch 9, bg 10-11, bk 12-13, col 14-18, row 19-20
  pim_friendly (base 0x40000000)  col 5-9, ch 10-13, pch 14, bg 15-16, bk 17-18, row 19-20
  joint        (base 0x80000000)  bg 5 and 11, ch 6-9, pch 10, bk 12-13, col 14-18, row 19-20

Row = (page number * 4) + the layout's 2 row bits, where page number = address / 2 MB:

  0x40000000 / 2 MB = page 512,  and 512 * 4 = row 2048
  0x80000000 / 2 MB = page 1024, and 1024 * 4 = row 4096
"""

import os
import subprocess
import tempfile
from pathlib import Path

import yaml

from scripts.make_region_config import OUT_PATH as REGION_CONFIG_PATH

DEFAULT_RAMULATOR = Path.home() / "build" / "ramulator2" / "ramulator2"
LEVELS = ("channel", "pseudochannel", "bankgroup", "bank", "row", "column")

# (address, region, levels that are not 0). Any level not listed must be 0.
CASES = [
    # npu_friendly: one bit at a time, then all of them together.
    (0x0, "npu_friendly", {}),
    (0x1F, "npu_friendly", {}),                       # Bits 0-4 select no level
    (0x20, "npu_friendly", {"channel": 1}),           # Bit 5, lowest ch bit
    (0x100, "npu_friendly", {"channel": 8}),          # Bit 8, highest ch bit
    (0x200, "npu_friendly", {"pseudochannel": 1}),    # Bit 9
    (0x400, "npu_friendly", {"bankgroup": 1}),        # Bit 10
    (0x800, "npu_friendly", {"bankgroup": 2}),        # Bit 11
    (0x1000, "npu_friendly", {"bank": 1}),            # Bit 12
    (0x2000, "npu_friendly", {"bank": 2}),            # Bit 13
    (0x4000, "npu_friendly", {"column": 1}),          # Bit 14
    (0x40000, "npu_friendly", {"column": 16}),        # Bit 18
    (0x80000, "npu_friendly", {"row": 1}),            # Bit 19
    (0x100000, "npu_friendly", {"row": 2}),           # Bit 20
    (0x200000, "npu_friendly", {"row": 4}),           # Bit 21, so page 1 and row 1 * 4
    (0x1FFFE0, "npu_friendly",                        # Bits 5-20 all set
     {"channel": 15, "pseudochannel": 1, "bankgroup": 3, "bank": 3, "row": 3, "column": 31}),

    # pim_friendly: every address here is at least page 512, so row 2048.
    (0x40000000, "pim_friendly", {"row": 2048}),
    (0x40000020, "pim_friendly", {"row": 2048, "column": 1}),         # Bit 5
    (0x40000400, "pim_friendly", {"row": 2048, "channel": 1}),        # Bit 10
    (0x40004000, "pim_friendly", {"row": 2048, "pseudochannel": 1}),  # Bit 14
    (0x40008000, "pim_friendly", {"row": 2048, "bankgroup": 1}),      # Bit 15
    (0x40020000, "pim_friendly", {"row": 2048, "bank": 1}),           # Bit 17
    (0x40080000, "pim_friendly", {"row": 2049}),                      # Bit 19, 2048 + 1
    (0x40200000, "pim_friendly", {"row": 2052}),                      # Page 513, 513 * 4

    # joint: every address here is at least page 1024, so row 4096.
    (0x80000000, "joint", {"row": 4096}),
    (0x80000020, "joint", {"row": 4096, "bankgroup": 1}),      # Bit 5, lower bg bit
    (0x80000800, "joint", {"row": 4096, "bankgroup": 2}),      # Bit 11, upper bg bit
    (0x80000040, "joint", {"row": 4096, "channel": 1}),        # Bit 6
    (0x80000400, "joint", {"row": 4096, "pseudochannel": 1}),  # Bit 10
    (0x80001000, "joint", {"row": 4096, "bank": 1}),           # Bit 12
    (0x80004000, "joint", {"row": 4096, "column": 1}),         # Bit 14
    (0x80080000, "joint", {"row": 4097}),                      # Bit 19, 4096 + 1
]


def run_ramulator(addresses, workdir):
    """Run the addresses through Ramulator with debug_decode on, and return its stderr."""
    trace_path = Path(workdir) / "mapper_test.trace"
    trace_path.write_text("".join(f"LD {address:#x}\n" for address in addresses))

    with open(REGION_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["Frontend"]["path"] = str(trace_path)
    config["MemorySystem"]["AddrMapper"]["debug_decode"] = True

    config_path = Path(workdir) / "mapper_test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    # RAMULATOR_BIN lets the test run against a build somewhere else.
    ramulator = os.environ.get("RAMULATOR_BIN", DEFAULT_RAMULATOR)
    result = subprocess.run([str(ramulator), "-f", str(config_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ramulator failed:\n{result.stderr}")
    return result.stderr


def parse_decodes(output):
    """Return address -> (region, levels) from lines such as:

    decode 0x20 npu_friendly channel=1 pseudochannel=0 bankgroup=0 bank=0 row=0 column=0

    Ramulator retries a request when memory is busy, so an address can print
    more than once. Each retry decodes the same way, so keeping one is enough.
    """
    decodes = {}
    for line in output.splitlines():
        if line.startswith("decode "):
            _, address, region, *fields = line.split()
            levels = {name: int(value) for name, value in (field.split("=") for field in fields)}
            decodes[int(address, 16)] = (region, levels)
    return decodes


def find_failures(cases, decodes):
    """Return one message for every case that does not match its hand-worked values."""
    failures = []
    for address, region, nonzero in cases:
        if address not in decodes:
            failures.append(f"{address:#x}: never decoded")
            continue
        # Fill in the 0s so the whole result is compared, not only the listed levels.
        expected = {level: nonzero.get(level, 0) for level in LEVELS}
        actual_region, actual = decodes[address]
        if actual_region != region or actual != expected:
            failures.append(f"{address:#x}: expected {region} {expected}, got {actual_region} {actual}")
    return failures


if __name__ == "__main__":
    # The trace and its config are throwaway, so they go in a temporary folder.
    with tempfile.TemporaryDirectory() as workdir:
        output = run_ramulator([case[0] for case in CASES], workdir)

    failures = find_failures(CASES, parse_decodes(output))
    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(f"{len(failures)} of {len(CASES)} addresses decoded wrongly")
    print(f"all {len(CASES)} addresses match the hand-worked values")