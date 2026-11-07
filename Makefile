# Run from the repository root.  `make help` lists the targets.
PY ?= python3
export PYTHONPATH := src
REPS ?= 30

.PHONY: help test figures paper thai simulate robust all clean

help:
	@echo "make test      unit tests (about 10 s)"
	@echo "make figures   rebuild the 8 figures from results/ with the automated overlap check"
	@echo "make paper     rebuild paper/arxiv/main.pdf (pdflatex)"
	@echo "make thai      rebuild the Thai draft paper/thai/paper_th.pdf (xelatex)"
	@echo "make simulate  rerun the Monte Carlo study, REPS=30 (about 15 min on one core)"
	@echo "make robust    rerun the robustness experiments R1 and R2 (about 2 min)"
	@echo "make all       simulate + robust + figures + paper"

test:
	$(PY) -m pytest -q tests

figures:
	$(PY) -c "import figs_en; figs_en.make_all(res='results', out='figures')"
	cp figures/*.png paper/arxiv/
	for f in fig_data fig_drivers fig_e1_paths fig_lambda fig_beta fig_e2; do cp figures/$$f.png paper/thai/; done

paper:
	$(PY) scripts/build_arxiv.py

thai:
	cd paper/thai && xelatex -interaction=nonstopmode paper_th.tex && xelatex -interaction=nonstopmode paper_th.tex

simulate:
	rm -f results/metrics.csv results/betas.csv results/lamhat.csv results/paths_seed0.pkl
	$(PY) src/hesi_forecast.py simulate --reps $(REPS) --out results

robust:
	rm -f results/robust_metrics.csv results/robust_betas.csv
	$(PY) src/robust_en.py 0 $(REPS) results

all: simulate robust figures paper

clean:
	rm -f paper/*/*.aux paper/*/*.log paper/*/*.out paper/*/*.toc
