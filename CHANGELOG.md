# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-11-07

### Added
- Holiday-Equivalent Stoppage (HES) framework: one shared stoppage regressor per channel (Thai public holidays
  including Songkran; the Chinese New Year window) in Prophet, carrying a pandemic dose, the HESI.
- Monte Carlo study with 30 replications: simulated export values driven by the real Thai holiday calendar,
  OxCGRT stringency and Google Community Mobility; experiments E0 (normal times), E1 (shock), E2 (after the shock).
- Robustness experiments R1 (heterogeneous holidays) and R2 (real-time, trailing HESI).
- Theory checks: aggregation attenuation to `exp(beta) - 1` and the closed-form aggregation correction.
- Installable package `hesi` (`pyproject.toml`, `make setup` with uv) and one command line, `hesi`
  (`simulate`, `robust`, `figures`, `paper`, `thai`, `web`, `real`), including a runner for official monthly
  export data (`hesi real`).
- The preprint, `paper/paper.pdf` (arXiv layout), built from `paper/template.tex` with tectonic or pdflatex, every
  table and in-text number filled from `results/`; a Thai-language draft, `paper/thai/paper_th.pdf`.
- Web reader of the paper, `docs/paper/index.html` (`make web`), built with pandoc from the same filled LaTeX as
  the PDF, served at <https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/>.
- arXiv source bundle, `paper/arxiv.tar.gz` (`make arxiv`): the staged copy is compiled once to prove it is
  self-contained, and the title, abstract and comments for the submission form are printed.
- Dataset on Hugging Face, [`thanatchon36/lockdowns-as-partial-holidays`](https://huggingface.co/datasets/thanatchon36/lockdowns-as-partial-holidays):
  daily drivers and shared regressors, the OxCGRT and Google Mobility extracts, simulated exports and forecast
  paths of replication 0, and all Monte Carlo results, as CSV with a dataset card, licence and checksums
  (`make dataset`).
- README banner, `docs/assets/banner.png` (`make banner`): the Thai Holiday-Equivalent Stoppage Index from the real
  Google Mobility data.
- Research prompt for replicating the study with official data (`prompts/research_prompt.md`).
- `CITATION.cff` with a preferred citation, `.zenodo.json` for the Zenodo archive, CI on GitHub Actions.

[Unreleased]: https://github.com/thanatchon36/lockdowns-as-partial-holidays/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/thanatchon36/lockdowns-as-partial-holidays/releases/tag/v1.0.0
