#!/usr/bin/env python3
# N1_philox.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N1: the Philox4x32-10 generator of rebar reproduces the Random123 known answers.
#
# This program was designed and developed by the author(s),
# with the assistance of the following AI tool(s):
#   2026 Claude (Anthropic).
# The final logic and implementation were reviewed and verified in
# their entirety by the author(s).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""N1 -- the Philox4x32-10 generator reproduces the Random123 known answers.

Claim (paper, Sec. II C): the random numbers are Philox4x32-10 (Salmon et al., SC'11), one key per class; the
implementation shared by the CPU and the GPU kernel (integrator/include/rbar/philox.h) reproduces the known-answer
vectors of Random123 (tests/kat_vectors, philox4x32 10 rounds; github.com/DEShawResearch/random123, commit
9545ff6413f258be2f04c1d319d99aaef7521150).

Inputs: none (the three Random123 vectors are transcribed below).
What runs: `rebar_cpu check --what philox` (the binary compares its outputs with the vectors itself), and an
independent Python implementation of Philox4x32-10 (written here from the published algorithm) on the same inputs.
PASS: the binary prints the three expected outputs and reports no mismatch, and the Python implementation gives the
same three outputs.
"""
import os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the Philox generator reproduces the Random123 known answers (Sec. II C)"
QUICK = True
# (counter, key, expected output), Random123 kat_vectors, philox4x32 10
KAT = [((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
       ((0xffffffff,) * 4, (0xffffffff,) * 2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
       ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0), (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))]
M0, M1, W0, W1, MASK = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85, 0xffffffff


def philox4x32_10(ctr, key):
    """Philox4x32 with 10 rounds: per round two 32x32 -> 64-bit products, the key bumped by the Weyl constants"""
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for r in range(10):
        if r:
            k0, k1 = (k0 + W0) & MASK, (k1 + W1) & MASK
        p0, p1 = M0 * c0, M1 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & MASK, p1 & MASK, ((p0 >> 32) ^ c3 ^ k1) & MASK, p0 & MASK
    return c0, c1, c2, c3


def main():
    py = [philox4x32_10(c, k) for c, k, _ in KAT]
    okpy = all(p == r for p, (_, _, r) in zip(py, KAT))
    for (c, k, r), p in zip(KAT, py):
        print("python   ctr %s key %s -> %s  %s" % (" ".join("%08x" % x for x in c), " ".join("%08x" % x for x in k),
                                                    " ".join("%08x" % x for x in p), "OK" if p == r else "MISMATCH"))
    exe = C.cpu_binary()
    if not exe:
        C.verdict("N1", okpy, "Python implementation %s the 3 vectors" % ("reproduces" if okpy else "does NOT reproduce"),
                  skip="the CPU binary (make -C integrator cpu VARIANTS=e16 failed)")
    st, so, se = C.run([exe, "check", "--what", "philox"])
    print("rebar_cpu check --what philox (exit status %d):\n%s" % (st, so.rstrip()))
    got = [tuple(int(x, 16) for x in m.groups()) for m in re.finditer(r"philox KAT: (\w+) (\w+) (\w+) (\w+)\s+OK", so)]
    okbin = st == 0 and got == [r for _, _, r in KAT]
    C.verdict("N1", okbin and okpy, "rebar_cpu: %d/3 known answers; independent Python Philox4x32-10: %d/3"
              % (sum(1 for g, (_, _, r) in zip(got, KAT) if g == r), sum(1 for p, (_, _, r) in zip(py, KAT) if p == r)))


if __name__ == "__main__":
    main()
