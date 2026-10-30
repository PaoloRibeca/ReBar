# Makefile -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Runs the reproducibility checks of the paper's claims (README.md): 'make' runs them all and writes REPORT.md,
# 'make quick' the quick ones, 'make G2' (or any id) one of them; 'make clean' removes the helpers, the cache of
# their outputs and the logs (checks/build), not REPORT.md.
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
#
#   make             all the checks -> REPORT.md (about 40 minutes on a 4-core workstation the first time, 30 once cached)
#   make quick       the checks marked quick
#   make G1 ... S5   one check (its row of REPORT.md is updated)
#   make list        the checks and their claims
#   make clean       rm -rf build
#
# Variables: PYTHON (>= 3.6, standard library only), REBAR_THREADS (OpenMP threads of the binaries, default 2),
# REBAR_JOBS (Python worker processes, default 2), REBAR_MPMATH_PYTHON (an interpreter with mpmath, for I1).

PYTHON ?= python3
IDS     = G1 G2 G3 G4 G5 I1 I2 I3 I4 N1 N2 N3 N4 N5 N6 N7 N8 S1 S2 S3 S4 S5

all:
	$(PYTHON) -I run.py

quick:
	$(PYTHON) -I run.py --quick

list:
	@$(PYTHON) -I run.py --list

$(IDS):
	$(PYTHON) -I run.py $@

clean:
	rm -rf build

.PHONY: all quick list clean $(IDS)
