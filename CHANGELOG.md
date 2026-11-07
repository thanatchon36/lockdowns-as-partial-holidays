# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Installable package `hesi` (`pyproject.toml`, `make setup` with uv) and one command line, `hesi`
  (`simulate`, `robust`, `figures`, `paper`, `thai`, `web`, `real`).
- Web reader of the paper, `docs/paper/index.html` (`make web`), built with pandoc from the same filled LaTeX as
  the PDF, for GitHub Pages at <https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/>.
- arXiv source bundle, `paper/arxiv.tar.gz` (`make arxiv`): the staged copy is compiled once to prove it is
  self-contained, and the title, abstract and comments for the submission form are printed.
- README banner, `docs/assets/banner.png` (`make banner`): the Thai Holiday-Equivalent Stoppage Index from the real
  Google Mobility data.
- `CHANGELOG.md`, `.zenodo.json` and a `preferred-citation` in `CITATION.cff`.

### Changed
- Layout: `src/*.py` became the package `src/hesi/` (`forecast`, `robust`, `figures`, `report`, `paper_build`);
  data moved to `data/raw/`, figures to `paper/figures/`, the paper source to `paper/template.tex` (built into
  `paper/paper.tex` and `paper/paper.pdf`), and `data/README.md` to `docs/data_sources.md`.
- The paper builds with tectonic (pdflatex as fallback); the template still compiles with pdflatex for arXiv.
- CI (`.github/workflows/ci.yml`) installs with uv through `make setup`, with actions pinned by commit.
- Dependencies moved from `requirements.txt` to `pyproject.toml`, with the same pins.

## [1.0.0] - 2026-11-07

### Added
- Holiday-Equivalent Stoppage (HES) framework: one shared stoppage regressor per channel (Thai public holidays
  including Songkran; the Chinese New Year window) in Prophet, carrying a pandemic dose, the HESI.
- Monte Carlo study with 30 replications: simulated export values driven by the real Thai holiday calendar,
  OxCGRT stringency and Google Community Mobility; experiments E0 (normal times), E1 (shock), E2 (after the shock).
- Robustness experiments R1 (heterogeneous holidays) and R2 (real-time, trailing HESI).
- Theory checks: aggregation attenuation to `exp(beta) - 1` and the closed-form aggregation correction.
- Runner for official monthly export data (`real`).
- The preprint (`paper.pdf`, arXiv layout), a Thai-language draft (`paper/thai/paper_th.pdf`) and a research
  prompt for replicating the study with official data (`prompts/research_prompt.md`).
