"""Build the arXiv source bundle, ``paper/arxiv.tar.gz``, from the paper as last built.

arXiv compiles the paper itself with pdfLaTeX (its default). This stages ``paper.tex`` (every number and table
already filled in), ``arxiv.sty`` and the figures it uses, compiles the staged copy once with tectonic to prove it
is self-contained, then tars it. It also prints the title, the plain-text abstract (for the arXiv form) and a
comments line. Run ``make paper`` first.

Usage: ``python scripts/arxiv_bundle.py``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE = PAPER / "arxiv"
OUT = PAPER / "arxiv.tar.gz"


def _plain(tex: str) -> str:
    """Rough LaTeX -> plain text for the arXiv abstract field (arXiv accepts $...$ math)."""
    tex = re.sub(r"\\noindent\s*", "", tex)
    tex = re.sub(r"\\(?:emph|textbf|textit)\{([^}]*)\}", r"\1", tex)
    tex = tex.replace("---", "--").replace("\\%", "%")
    return " ".join(tex.split())


def main() -> int:
    tex = (PAPER / "paper.tex").read_text(encoding="utf-8")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "figures").mkdir(parents=True)
    (STAGE / "paper.tex").write_text(tex, encoding="utf-8")
    shutil.copyfile(PAPER / "arxiv.sty", STAGE / "arxiv.sty")
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([\w.-]+)\}", tex)))
    for name in figures:
        shutil.copyfile(PAPER / "figures" / name, STAGE / "figures" / name)
    # the staged copy must compile on its own
    subprocess.run(["tectonic", "-X", "compile", "--keep-logs", "paper.tex"], cwd=STAGE, check=True,
                   capture_output=True)
    log = (STAGE / "paper.log").read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on paper\.(?:xdv|pdf) \((\d+) pages", log)
    if pages is None:
        raise SystemExit("staged compile left no page count")
    for name in ("paper.log", "paper.pdf"):
        (STAGE / name).unlink(missing_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(STAGE)))
    files = sorted(str(p.relative_to(STAGE)) for p in STAGE.rglob("*") if p.is_file())
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} kB, {len(files)} files)")
    print("\n".join(f"  {f}" for f in files))

    title = re.search(r"\\title\{(.*?)\}\n", tex, re.S)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if title is None or abstract is None:
        raise SystemExit("paper.tex lacks the title or abstract")
    body = _plain(abstract.group(1))
    print(f"\nTitle:\n{_plain(title.group(1).replace(chr(92) * 2, ' '))}")
    print(f"\nAbstract ({len(body)} characters; arXiv allows 1,920):\n{body}")
    print(f"\nComments:\n{pages.group(1)} pages, {len(figures)} figures. Monte Carlo study: export values are simulated; "
          "holiday calendar, OxCGRT and Google Mobility data are real. "
          "Code: https://github.com/thanatchon36/lockdowns-as-partial-holidays. JEL C15, C22, C53, F17.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
