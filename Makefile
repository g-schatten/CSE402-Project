.PHONY: setup core experiments experiment figures quicklook fixtures web web-build web-test report slides all clean test

PY := . .venv/bin/activate &&
PROFILE ?= quick

setup:
	python3 -m venv .venv
	$(PY) pip install -q --upgrade pip
	$(PY) pip install -q -e ".[dev]"
	cd web && npm install

test:
	$(PY) pytest tests/ -q

core: test
	@echo "core library tests passed"

# Run every experiment (quick profile by default; PROFILE=full for the paper run)
experiments:
	$(PY) python -m experiments.runner --all --profile $(PROFILE)
	mkdir -p web/public/data
	cp results/*.json web/public/data/ 2>/dev/null || true

# Run a single experiment: make experiment ID=E07 PROFILE=full
experiment:
	$(PY) python -m experiments.runner $(ID) --profile $(PROFILE)
	mkdir -p web/public/data
	cp results/$(ID).json web/public/data/ 2>/dev/null || true

# Figures used by the report (report/figures/*.pdf)
figures:
	$(PY) python report/make_report_figures.py

# Quick-look figures of every experiment (figures/, git-ignored)
quicklook:
	$(PY) python scripts/make_figures.py

fixtures:
	$(PY) python scripts/gen_fixtures.py

web:
	cd web && npm run dev

web-test: fixtures
	cd web && npx vitest run

web-build:
	cd web && npm run build

# Final report (needs a local TeX install; or upload report/ to Overleaf)
report: figures
	cd report && latexmk -pdf B_02.tex

# Figures for the supervisor slides (presentations/supervisor/figures/)
slides:
	$(PY) python presentations/supervisor/make_figures.py

all: setup core experiments web-test figures

clean:
	rm -rf results/*.json web/public/data/*.json figures/*.pdf figures/*.svg
	cd report && latexmk -c B_02.tex 2>/dev/null || true
