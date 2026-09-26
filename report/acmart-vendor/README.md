# Vendored acmart

This directory holds the `acmart` LaTeX class (v2.20, from CTAN,
<https://ctan.org/pkg/acmart>), its bibliography style
(`ACM-Reference-Format.bst`), and the handful of CTAN packages it depends on
that are **not** part of a default TeX Live install: `xstring`, `ifmtarg`,
`comment`, `environ`, `trimspaces`, `manyfoot`/`nccfoots` (from the
`ncctools` bundle), `refcount`, `totpages`, `hyperxmp`, `authblk`, and
`balance`.

They are vendored here — rather than assumed to be installed system-wide —
so that `make report` / `latexmk -pdf main.tex` builds the PDF on a clean
clone with no `sudo`, `tlmgr`, or internet access required.
`report/.latexmkrc` adds this directory to `TEXINPUTS`/`BSTINPUTS` for
exactly that purpose.

acmart is distributed under the LaTeX Project Public License (LPPL); see
<https://ctan.org/pkg/acmart> for the full license and documentation
(`acmart.pdf`, `acmguide.pdf` in the upstream distribution).

If your TeX installation already ships `acmart` (e.g. a full TeX Live or
Overleaf), this directory is redundant but harmless.
