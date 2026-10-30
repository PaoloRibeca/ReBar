# Makefile -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Builds bin/topologies, the generator of the Feynman graphs with a given list of vertex valences (the
# vertex-insertion generator of 2026, which derives from the generator of 2000-2003 behind GAU and the
# seven-loop computation), and checks it against the stored lists: 'make check' regenerates the phi^4 lists
# through 8 loops (two-point) and 7 loops (four-point), compares them with lists as sets of graphs and,
# relabelled with -G, byte for byte where the stored labelling allows it, checks the zero-dimensional sum
# rules and the integrator's lists, runs the Python checks of tools/ and compares the general mode with a
# brute-force enumeration on small degree lists; 'make check-full' goes through 10 loops (two-point) and
# 9 loops (four-point).
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
#   make              bin/topologies
#   make check        the phi^4 lists through 8 loops (two-point) and 7 loops (four-point) against lists and
#                     the sum rules, the integrator's lists, the Python checks, the general mode against the
#                     brute force (a few minutes)
#   make check-full   through 10 loops (two-point) and 9 loops (four-point): about ten minutes on two threads
#   make clean
#
# Variables: CXX (C++17; g++ 7.5 and 16.2 build it without warnings), CXXFLAGS, PYTHON (>= 3.6, standard
# library only), POPCNT (-mpopcnt; empty on a CPU without the instruction), THREADS (the OpenMP threads of
# the checks, default 2). The generator is one translation unit, topologies/topologies.cpp, which includes
# the headers of topologies/; it uses OpenMP.

CXX      ?= g++
CXXSTD    = -std=c++17
CXXFLAGS ?= -O2 -Wall -Wextra
LDFLAGS  ?=
PYTHON   ?= python3
THREADS  ?= 2

# The colour refinement counts bits: without the popcount instruction (x86-64-v2, every x86 CPU since 2008)
# the library routine makes the program four times slower
POPCNT ?= -mpopcnt

bin/topologies: topologies/topologies.cpp $(wildcard topologies/*.hpp)
	@mkdir -p bin
	$(CXX) $(CXXSTD) $(CXXFLAGS) $(POPCNT) -fopenmp -o $@ topologies/topologies.cpp $(LDFLAGS)

check: bin/topologies
	PYTHON=$(PYTHON) scripts/check.sh bin/topologies build/check 8 8 $(THREADS)

check-full: bin/topologies
	PYTHON=$(PYTHON) scripts/check.sh bin/topologies build/check 10 10 $(THREADS) 9

clean:
	rm -rf bin build

.PHONY: check check-full clean
