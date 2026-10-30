#!/bin/bash
# arxiv.sh -- assemble the arXiv submission package of the paper and build it the way arXiv does.
#
#   bash arxiv.sh          writes arxiv/ReBar-paper.tar.gz (and arxiv/abstract.txt for the submission form)
#
# arXiv compiles the sources itself with pdflatex, without bibtex and without our TEXINPUTS, so the package holds
# main.tex with sections/ and generated/, the bibliography as main.bbl (references.bib too, for readers), and the
# bundled REVTeX 4.2 files at its top level.  The package is built in a scratch directory with plain pdflatex, three
# passes, and must give the same text and the same number of pages as make's main.pdf; the abstract has every
# generated macro replaced by its value.  Needs bash, gawk, pdflatex, pdftotext, pdfinfo and an up-to-date main.pdf
# (make).  The output directory arxiv/ is ignored by git.
set -eu
cd "$(dirname "$0")"
[ -f main.bbl ] && [ -f main.pdf ] || { echo "arxiv.sh: build the paper first (make)" >&2; exit 1; }
[ main.pdf -nt main.tex ] && [ main.pdf -nt references.bib ] || { echo "arxiv.sh: main.pdf is older than the sources: make first" >&2; exit 1; }
for f in sections/*.tex generated/*.tex; do [ main.pdf -nt "$f" ] || { echo "arxiv.sh: main.pdf is older than $f: make first" >&2; exit 1; }; done
out=arxiv; pkg=$out/ReBar-paper
rm -rf "$pkg" "$pkg.tar.gz"; mkdir -p "$pkg/sections" "$pkg/generated"
cp main.tex main.bbl references.bib "$pkg/"
cp sections/*.tex "$pkg/sections/"; cp generated/*.tex "$pkg/generated/"
cp tex/*.cls tex/*.rtx tex/*.sty tex/*.bst "$pkg/"
# The abstract for the submission form: the text between \begin{abstract} and \end{abstract}, with the generated
# macros (\newcommand{\name}{value}, with or without *) replaced by their values, joined into one paragraph.
gawk '
  FNR == 1 && FILENAME ~ /generated\// { ingen = 1 } FNR == 1 && FILENAME !~ /generated\// { ingen = 0 }
  ingen {   # every \newcommand{\name}{value} (or \newcommand*) of the line, braces balanced
    line = $0
    while (match(line, /\\newcommand\*?\{\\([A-Za-z]+)\}\{/, m)) {
      rest = substr(line, RSTART + RLENGTH); depth = 1; val = ""
      for (i = 1; i <= length(rest) && depth > 0; i++) { c = substr(rest, i, 1); if (c == "{") depth++; else if (c == "}") depth--; if (depth > 0) val = val c }
      macro[m[1]] = val; line = substr(rest, i) }
    next }
  !ingen && /\\begin\{abstract\}/ { inab = 1; next }
  !ingen && /\\end\{abstract\}/ { inab = 0 }
  !ingen && inab {
    line = $0; sub(/[ \t]*%.*$/, "", line)
    macro["Rbar"] = "\\bar R"; macro["ubar"] = "\\bar u"
    out = ""
    while (match(line, /\\([A-Za-z]+)(\\ |\{\})?/, m)) {
      pre = substr(line, 1, RSTART - 1); hit = substr(line, RSTART, RLENGTH); line = substr(line, RSTART + RLENGTH)
      if (m[1] in macro) out = out pre macro[m[1]] ((m[2] == "\\ ") ? " " : ""); else out = out pre hit }
    line = out line
    text = text (text == "" ? "" : " ") line }
  END {
    gsub(/\\,/, ",", text)          # thin spaces of numbers -> thousands separators
    gsub(/---/, " -- ", text); text = gensub(/([0-9])--([0-9])/, "\\1-\\2", "g", text)
    gsub(/  +/, " ", text); print text }
' generated/*.tex sections/abstract.tex > "$out/abstract.txt"
# Build as arXiv does: plain pdflatex, three passes, no bibtex, no TEXINPUTS.
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp -r "$pkg" "$tmp/build"
( cd "$tmp/build" && for i in 1 2 3; do pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true; done )
[ -f "$tmp/build/main.pdf" ] || { echo "arxiv.sh: the package does not build; see $tmp/build/main.log" >&2; trap - EXIT; exit 1; }
if grep -q "undefined\|LaTeX Error\|Emergency stop" "$tmp/build/main.log"; then
  echo "arxiv.sh: the package builds with errors or undefined references:" >&2
  grep -n "undefined\|LaTeX Error\|Emergency stop" "$tmp/build/main.log" | head -5 >&2; trap - EXIT; exit 1
fi
pages=$(pdfinfo "$tmp/build/main.pdf" | gawk '/^Pages:/ { print $2 }')
ours=$(pdfinfo main.pdf | gawk '/^Pages:/ { print $2 }')
[ "$pages" = "$ours" ] || { echo "arxiv.sh: the package gives $pages pages, make's main.pdf $ours" >&2; exit 1; }
if ! cmp -s <(pdftotext "$tmp/build/main.pdf" - | grep -v "Dated:") <(pdftotext main.pdf - | grep -v "Dated:"); then
  echo "arxiv.sh: the text of the package's PDF differs from make's main.pdf:" >&2
  diff <(pdftotext "$tmp/build/main.pdf" -) <(pdftotext main.pdf -) | head -10 >&2; exit 1
fi
( cd "$out" && tar czf ReBar-paper.tar.gz ReBar-paper )
echo "arxiv.sh: $pkg.tar.gz ($(du -h "$pkg.tar.gz" | cut -f1), $pages pages, same text as main.pdf); abstract in $out/abstract.txt"
