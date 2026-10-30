"""Generate a read trace that keeps every pseudo-channel busy.

Address bits follow Ramulator's RoBaRaCoCh mapper for configs/hbm2e_pfm.yaml:
  offset [4:0], channel [8:5], column [13:9], pseudochannel [14],
  bankgroup [16:15], bank [18:17], row [33:19]

Request order (fastest first): channel, pseudochannel, bankgroup, column,
bank, row. Consecutive reads to a pseudo-channel switch bank group, so they
are spaced by tCCD_S instead of tCCD_L, and each row is read fully once
opened.
"""

import argparse

CHANNELS = 16
PSEUDOCHANNELS = 2
BANKGROUPS = 4
COLUMNS = 32
BANKS = 4


def address(i):
    """Return the address of request number i.

    The fields count like the digits of an odometer. Channel is the fastest
    digit; each slower digit moves only when the one before it wraps:

        channel        changes every request
        pseudochannel  changes every 16 requests
        bankgroup      changes every 32 requests (16 * 2)
        column         changes every 128 requests (32 * 4)
        bank           changes every 4096 requests (128 * 32)
        row            changes every 16384 requests (4096 * 4)

    Generation order:
      1. Visit all 16 channels in pseudochannel 0.
      2. Repeat for pseudochannel 1.
      3. Move to the next bankgroup and repeat steps 1-2.
      4. After all 4 bankgroups, move to the next column and start again.

    Example, i = 49:
      channel:       49 % 16 = 1, then 49 // 16 = 3   -> ch = 1
      pseudochannel: 3 % 2 = 1, then 3 // 2 = 1       -> pch = 1
      bankgroup:     1 % 4 = 1, then 1 // 4 = 0       -> bg = 1
      the rest:      only 0 is left                   -> col = ba = row = 0
      address:       (1 << 5) | (1 << 14) | (1 << 15)
                     = 0x20 | 0x4000 | 0x8000 = 0xc020

    First addresses generated:

      request  address  ch  pch  bg  col  what changed
            0  0x0       0    0   0    0  start
            1  0x20      1    0   0    0  next channel
           15  0x1e0    15    0   0    0  last channel
           16  0x4000    0    1   0    0  channels wrapped, pseudochannel moves
           31  0x41e0   15    1   0    0  last channel again
           32  0x8000    0    0   1    0  pseudochannel wrapped, bankgroup moves
           48  0xc000    0    1   1    0  pseudochannel moves
           64  0x10000   0    0   2    0  bankgroup moves
           96  0x18000   0    0   3    0  bankgroup moves
          127  0x1c1e0  15    1   3    0  everything at column 0 is done
          128  0x200     0    0   0    1  bankgroup wrapped, column moves
    """
    # % reads the current digit, //= drops it so the next digit is in place.
    ch = i % CHANNELS
    i //= CHANNELS
    pch = i % PSEUDOCHANNELS
    i //= PSEUDOCHANNELS
    bg = i % BANKGROUPS
    i //= BANKGROUPS
    col = i % COLUMNS
    i //= COLUMNS
    ba = i % BANKS
    row = i // BANKS

    # Move each field to its bit position in the mapper's layout. This differs
    # from the counting order above: column bits sit below pseudochannel and
    # bankgroup, so the addresses do not increase steadily. For example, the
    # address drops from 0x1c1e0 to 0x200 at request 128.
    return (ch << 5) | (col << 9) | (pch << 14) | (bg << 15) | (ba << 17) | (row << 19)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--num-requests", type=int, default=1_000_000)     # generate 1 million addresses
    parser.add_argument("-o", "--output", default="traces/peak.trace")
    args = parser.parse_args()

    with open(args.output, "w") as f:
        for i in range(args.num_requests):
            f.write(f"LD {address(i):#x}\n")


if __name__ == "__main__":
    main()