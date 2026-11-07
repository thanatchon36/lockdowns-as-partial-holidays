PY ?= .venv/bin/python
REPS ?= 30

.PHONY: setup test simulate robust figures paper thai web arxiv banner all clean

setup:          ## create venv and install the package (needs uv)
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -e ".[dev]"

test:           ## unit tests (about 10 s)
	$(PY) -m pytest -q

simulate:       ## Monte Carlo study, REPS=30 (about 15 min on one core) -> results/
	rm -f results/metrics.csv results/betas.csv results/lamhat.csv results/paths_seed0.pkl
	$(PY) -m hesi.cli simulate --reps $(REPS) --out results

robust:         ## robustness experiments R1 and R2 (about 2 min) -> results/robust_*.csv
	rm -f results/robust_metrics.csv results/robust_betas.csv
	$(PY) -m hesi.cli robust --reps $(REPS) --out results

figures:        ## the eight paper figures from results/ -> paper/figures/, with the automated overlap check
	$(PY) -m hesi.cli figures

paper:          ## paper/template.tex -> paper/paper.tex -> paper/paper.pdf (tectonic, or pdflatex), then the web reader
	$(PY) -m hesi.cli paper
	$(PY) -m hesi.cli web

thai:           ## Thai draft paper/thai/paper_th.pdf (tectonic, or xelatex)
	$(PY) -m hesi.cli thai

web:            ## web reader only -> docs/paper/index.html (pandoc; copies the current paper.pdf)
	$(PY) -m hesi.cli web

arxiv: paper    ## arXiv source bundle -> paper/arxiv.tar.gz
	$(PY) scripts/arxiv_bundle.py

banner:         ## README banner -> docs/assets/banner.png (real Google Mobility data)
	$(PY) scripts/make_banner.py

all: simulate robust figures paper

clean:
	rm -rf paper/paper.tex paper/paper.log paper/arxiv paper/arxiv.tar.gz paper/thai/*.log paper/*.aux paper/*.out
