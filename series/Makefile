# Makefile -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Regenerates what is derived from the ledger: the per-order sums and RG coefficients (generated/coefficients.*),
# the large-order test (generated/checks.*) and the paper's fragments (../paper/generated/values.tex, newsums.tex,
# validation.tex); 'make check' runs the self-tests (Phi4tools round trip, reproduction of the eight-loop numbers,
# pooling, merge rules and idempotency).
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
#   make             generated/coefficients.{json,txt}, generated/checks.{json,txt}, ../paper/generated/*.tex
#   make check       self-tests (about 20 s; the exact pooling test uses ../integrator/bin/rebar_cpu if it exists)
#   make status      the orders of the ledger
#   make rebuild     recompute the order statistics of ../ledger/orders.json (format 3: from the stats)
#   make sumrules    zero-dimensional sum rules of the lists of the ledger's orders (minutes at 8 loops)
#
# Variables: PYTHON (>= 3.6, standard library only).

PYTHON ?= python3
PY      = $(PYTHON) -I
LEDGER  = ../ledger/orders.json
LIB     = rebar_ledger.py
ROUTEB  = ../results/checks/routeb/runs
ROUTEB10 = ../results/checks/routeb10/runs
ROUTEB_RUNS = --run int4:7:$(ROUTEB)/int4_L7.json.xz --run int2:8:$(ROUTEB)/int2_L8.json.xz \
              --run int2d:8:$(ROUTEB)/int2d_L8.json.xz --run int4:8:$(ROUTEB)/int4_L8.json.xz \
              --run int2:9:$(ROUTEB)/int2_L9.json.xz --run int2d:9:$(ROUTEB)/int2d_L9.json.xz \
              --run int4:9:$(ROUTEB10)/int4_L9.json.xz --run int2:10:$(ROUTEB10)/int2_L10.json.xz \
              --run int2d:10:$(ROUTEB10)/int2d_L10.json.xz

all: paper

generated/coefficients.json: $(LEDGER) coefficients.py $(LIB) ../integrator/tools/refs.py
	@mkdir -p generated
	$(PY) coefficients.py --json $@ > generated/coefficients.txt

generated/checks.json: generated/coefficients.json checks.py $(LIB)
	$(PY) checks.py --json $@ > generated/checks.txt

generated/polynomials.json: $(LEDGER) polynomials.py coefficients.py tables.py $(LIB) ../integrator/tools/refs.py
	@mkdir -p generated
	$(PY) polynomials.py --tex ../paper/generated/newpoly.tex --json $@ --tsv generated/polynomials.tsv > generated/polynomials.txt

paper: generated/coefficients.json generated/checks.json generated/polynomials.json tables.py largeorder.py costfig.py routeb.py resum.py $(LIB)
	$(PY) tables.py
	$(PY) largeorder.py --N 0,1,2,3,4 --tex ../paper/generated/largeorder.tex --fit-tex ../paper/generated/largeorder-fit.tex > generated/largeorder.txt
	$(PY) costfig.py > generated/costhist.txt
	$(PY) resum.py --tex ../paper/generated/exponents.tex > generated/resum.txt
	$(PY) routeb.py --tex ../paper/generated/routes.tex $(ROUTEB_RUNS) > generated/routes.txt

check:
	$(PY) tests.py

status:
	@$(PY) rebar_ledger.py status

rebuild:
	$(PY) rebar-merge.py --rebuild

sumrules:
	$(PY) checks.py --sumrules

.PHONY: all paper check status rebuild sumrules
