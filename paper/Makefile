# Makefile -- build the paper with the bundled REVTeX 4.2 (./tex/)
#   make            two-column PRD layout   -> main.pdf
#   make onecolumn  single-column draft     -> main-onecolumn.pdf
#   make all        both
#   make pages      rasterise main.pdf to png/ (60 dpi) for a quick look at the layout
#   make clean      remove auxiliary files (keeps the PDFs)
#   make distclean  remove auxiliary files and PDFs
#
# The numbers of the paper are in generated/*.tex, written by `make -C ../series` from the ledger.

export TEXINPUTS := ./tex//:$(TEXINPUTS)
export BSTINPUTS := ./tex//:$(BSTINPUTS)

LATEXMK := latexmk -pdf -r latexmkrc
SOURCES := main.tex references.bib $(wildcard sections/*.tex) $(wildcard generated/*.tex) $(wildcard tex/*)

.PHONY: default all onecolumn pages clean distclean

default: main.pdf

all: main.pdf main-onecolumn.pdf

onecolumn: main-onecolumn.pdf

main.pdf: $(SOURCES)
	$(LATEXMK) main.tex

main-onecolumn.pdf: main-onecolumn.tex $(SOURCES)
	$(LATEXMK) main-onecolumn.tex

pages: main.pdf
	rm -rf png && mkdir -p png && pdftocairo -png -r 60 main.pdf png/page

clean:
	$(LATEXMK) -c main.tex
	$(LATEXMK) -c main-onecolumn.tex
	rm -f mainNotes.bib main-onecolumnNotes.bib *.run.xml
	rm -rf png

distclean: clean
	rm -f main.pdf main-onecolumn.pdf
