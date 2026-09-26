# Vendored copy of acmart (class, .bbx/.cbx, ACM-Reference-Format.bst) and its
# CTAN dependencies that are not part of a default TeX Live install
# (xstring, ifmtarg, comment, environ, trimspaces, manyfoot/nccfoots,
# refcount, totpages, hyperxmp, authblk, balance). This lets `latexmk -pdf
# main.tex` (and `make report`) succeed on a clean clone without installing
# anything system-wide; see acmart-vendor/README.md.
ensure_path('TEXINPUTS', './acmart-vendor//');
ensure_path('BSTINPUTS', './acmart-vendor//');
