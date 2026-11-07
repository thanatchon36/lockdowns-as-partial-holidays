"""Build the web reader docs/paper/ from the filled LaTeX source (same text and numbers as the PDF).

`paper_build.render()` writes paper/paper.tex; this module simplifies it for pandoc (its LaTeX reader stalls on
the arXiv preamble and has no TikZ or natbib), renders it with paper/reader.html, and copies the PDF and the
figures next to the page. GitHub Pages serves docs/ at https://thanatchon36.github.io/lockdowns-as-partial-holidays/.
Requires pandoc on PATH. Run from the repository root (`make web`).
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path

from . import paper_build as PB

REPO_URL = "https://github.com/thanatchon36/lockdowns-as-partial-holidays"
PAPER_URL = "https://thanatchon36.github.io/lockdowns-as-partial-holidays/paper/"
DATA_URL = "https://huggingface.co/datasets/thanatchon36/lockdowns-as-partial-holidays"
TITLE = "Lockdowns as Partial Holidays"
SUBTITLE = ("Identifying Work-Stoppage Elasticities from Recurring Holidays and Transferring Them to Forecast "
            "Thailand's Monthly Exports through the COVID-19 Shock")

# (label, style, finding, detail); <<key>> resolves from the same macros as the paper.
# style: "pos" = the method works, "ref" = qualified or mixed, "null" = it does not, or pending.
RESULTS = [
    ("Transfer · S1", "pos",
     "HESI dose cuts the 2020-21 MAPE from <<e1.s1.pd.off.w24>>% to <<e1.s1.pd.hesi.w24>>%",
     "daily Prophet, ideal transfer; whole-month switch <<e1.s1.pd.binary.w24>>%, raw stringency "
     "<<e1.s1.pd.stringency.w24>>%"),
    ("Transfer · S2", "ref",
     "Realistic scenario: <<e1.s2.pd.hesi.w24>>% with HESI against <<e1.s2.pd.off.w24>>% without a dose",
     "mobility overstates stoppage two-fold; the gain is not significant after Holm, but a dose calibrated "
     "on Mar-Apr 2020 is significantly better"),
    ("Aggregation", "pos",
     "Monthly models attenuate the elasticity: <<bD.pm>> against the predicted <<bM.closed>>",
     "daily estimate <<bD.pd>> (true <<bD.true>>); the closed-form correction recovers <<bD.pmac>>"),
    ("Tree ensembles", "null",
     "LightGBM cannot transfer a dose: MAPE <<e1.s1.ld.hesi.w24>>% (S1), <<e1.s2.ld.hesi.w24>>% (S2)",
     "any positive dose falls past the split at 1e-35 and becomes a full holiday"),
    ("After the shock", "ref",
     "SARIMAX <<e2.s1.sx.hesi>>% and masking the window <<e2.s1.pd.mask>>% match or beat HES",
     "2023-2024 forecasts; HES is a tool for measurable stoppage shocks, not a universal gain"),
    ("Thai export data", "null",
     "Official-data results pending",
     "export values in every number above are simulated; hesi real runs the same methods on official data"),
]


def _bib(body: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """thebibliography -> a References section; returns it and key -> (authors, year) from the natbib labels."""
    m = re.search(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", body, re.S)
    if not m:
        return body, {}
    refs, items = {}, []
    for label, key, text in re.findall(r"\\bibitem\[(.*?)\]\{([^}]+)\}(.*?)(?=\\bibitem\[|\Z)", m.group(1), re.S):
        lab = re.match(r"(.*)\(([^()]*)\)\s*$", label.replace("{", "").replace("}", ""))
        refs[key] = (lab.group(1).replace("~", " ").strip(), lab.group(2)) if lab else (label, "")
        items.append(f"\\item \\label{{ref-{key}}}{text.strip()}")
    section = "\\section*{References}\n\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}\n"
    return body[:m.start()] + section + body[m.end():], refs


def _cites(body: str, refs: dict[str, tuple[str, str]]) -> str:
    """\\citep{a,b} -> (A, 2001; B, 2002) and \\citet{a} -> A (2001), as plain text."""
    def one(key: str, textual: bool) -> str:
        a, y = refs.get(key.strip(), (key.strip(), ""))
        return f"{a} ({y})" if textual else f"{a}, {y}"

    body = re.sub(r"\\citet\{([^}]*)\}", lambda m: "; ".join(one(k, True) for k in m.group(1).split(",")), body)
    return re.sub(r"\\citep\{([^}]*)\}", lambda m: "(" + "; ".join(one(k, False) for k in m.group(1).split(",")) + ")",
                  body)


def _number_equations(body: str) -> tuple[str, dict[str, str]]:
    """Number equation/align rows as LaTeX does (\\tag{n} for MathML) and map each eq label to its number."""
    n, labels = 0, {}

    def env(m: re.Match) -> str:
        nonlocal n
        name, inner = m.group(1), m.group(2)
        rows = inner.split("\\\\") if name == "align" else [inner]
        out = []
        for row in rows:
            if "\\nonumber" in row or "\\notag" in row or not row.strip():
                out.append(row)
                continue
            n += 1
            lab = re.search(r"\\label\{([^}]*)\}", row)
            if lab:
                labels[lab.group(1)] = str(n)
            out.append(row.rstrip() + f" \\tag{{{n}}}\n" if row.endswith("\n") else row + f" \\tag{{{n}}}")
        return f"\\begin{{{name}}}" + "\\\\".join(out) + f"\\end{{{name}}}"

    return re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", env, body, flags=re.S), labels


def _simplify(tex: str) -> tuple[str, dict]:
    """Keep the macros pandoc understands, drop the arXiv preamble, and pull out the front matter."""
    pre, body = tex.split("\\begin{document}", 1)
    keep = [ln for ln in pre.splitlines() if ln.startswith(("\\newcommand", "\\DeclareMathOperator", "\\newtheorem",
                                                             "\\theoremstyle"))]
    meta = {"date": re.search(r"\\date\{([^}]*)\}", pre).group(1)}
    kw = re.search(r"\\keywords\{(.*?)\}\s*\n", body, re.S)
    meta["keywords"] = ", ".join(s.strip() for s in kw.group(1).split("\\and")) if kw else ""
    jel = re.search(r"\{\\bfseries\\emph\{JEL codes\}\}\\enspace\s*([^\n]+)", body)
    meta["jel"] = jel.group(1).strip() if jel else ""
    body = body.replace(kw.group(0), "") if kw else body
    body = re.sub(r"\\noindent\{\\bfseries\\emph\{JEL codes\}\}[^\n]*\n", "", body)
    body = body.replace("\\maketitle", "")
    body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", r"\\includegraphics{fig_framework.png}", body,
                  flags=re.S)
    body, refs = _bib(body)
    body = _cites(body, refs)
    body, meta["eqs"] = _number_equations(body)
    main, _, app = body.partition("\\appendix")
    meta["appendix"] = re.findall(r"\\section\{[^}]*\}\\label\{([^}]*)\}", app)
    return "\\documentclass{article}\n" + "\n".join(keep) + "\n\\begin{document}" + body, meta


def _number_captions(page: str) -> str:
    """Table and figure numbers in captions, in document order (the PDF numbers them the same way)."""
    count = {"caption": 0, "figcaption": 0}

    def caption(m: re.Match) -> str:
        tag, attrs, para = m.group(1), m.group(2), m.group(3) or ""
        count[tag] += 1
        word = "Table" if tag == "caption" else "Figure"
        return f"<{tag}{attrs}>{para}<b>{word} {count[tag]}.</b> "

    return re.sub(r"<(caption|figcaption)(\b[^>]*)>(\s*<p>)?", caption, page)


def _equation_number(m: re.Match) -> str:
    """MathML has no \\tag: show the numbers beside the equation and anchor its labels for the references."""
    math = m.group(0)
    tags = re.findall(r"\\tag\{(\d+)\}", math)
    if not tags:
        return math
    labels = re.findall(r"\\label\{([^}]*)\}", math)
    anchors = "".join(f'<span id="{html.escape(lab)}"></span>' for lab in labels)
    nums = "<br>".join(f"({t})" for t in tags)
    return f'<div class="eq">{anchors}{math}<span class="eqno">{nums}</span></div>'


def _fill(s: str) -> str:
    return re.sub(r"<<([^<>]+)>>", lambda m: PB.R[m.group(1)], s)


def build_web(paper_dir: Path = Path("paper"), out_dir: Path = Path("docs/paper")) -> Path:
    """paper.tex -> simplified LaTeX -> out_dir/index.html (pandoc, reader.html), plus paper.pdf and figures."""
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc not found; install pandoc to build the web reader")
    tex, meta = _simplify(Path(PB.render()).read_text(encoding="utf-8"))
    page = subprocess.run(
        ["pandoc", "--from=latex", "--to=html5", "--mathml", "--wrap=none", "--standalone",
         "--template=reader.html", "--number-sections", "--toc", "--toc-depth=1",
         f"--metadata=title:{TITLE}", f"--metadata=subtitle:{SUBTITLE}", f"--metadata=pagetitle:{TITLE}",
         "--metadata=author:Thanatchon Chongmankhong", f"--metadata=date:{meta['date']}",
         f"--metadata=keywords:{meta['keywords']}", f"--metadata=jel:{meta['jel']}",
         "--metadata=description:Preprint: lockdowns as partial holidays. Work-stoppage elasticities learned from "
         "recurring holidays forecast Thailand's monthly exports through the COVID-19 shock."],
        input=tex, capture_output=True, text=True, check=True, cwd=str(paper_dir),
    ).stdout

    page = _number_captions(page)
    # equation references: "[eq:x]" -> "(n)", numbered like the PDF
    page = re.sub(r'(data-reference="(eq:[^"]+)">)\[[^\]]*\]',
                  lambda m: m.group(1) + "(" + meta["eqs"].get(m.group(2), "?") + ")", page)
    # appendices are lettered (A, B, ...) as in the PDF
    for i, lab in enumerate(meta["appendix"]):
        letter = chr(ord("A") + i)
        num = re.search(rf'<h1 data-number="(\d+)" id="{re.escape(lab)}"', page)
        if not num:
            continue
        page = re.sub(rf'(id="{re.escape(lab)}"><span class="header-section-number">){num.group(1)}<',
                      rf"\g<1>{letter}<", page)
        page = re.sub(rf'(data-reference="{re.escape(lab)}">){num.group(1)}<', rf"\g<1>{letter}<", page)
        page = re.sub(rf'(href="#{re.escape(lab)}"[^>]*><span class="toc-section-number">){num.group(1)}<',
                      rf"\g<1>{letter}<", page)
    page = re.sub(r'<math display="block".*?</math>', _equation_number, page, flags=re.DOTALL)
    page = re.sub(r'src="(fig_[\w.-]+)"', r'src="figures/\1"', page)
    page = re.sub(r"<img\b([^>]*)>", lambda m: "<img loading=\"lazy\"" + re.sub(r'\s(?:style|width|height)="[^"]*"',
                                                                               "", m.group(1)) + ">", page)
    page = re.sub(r"<table\b[^>]*>\s*<caption>(.*?)</caption>(.*?)</table>",
                  r'<div class="tbl"><p class="tcap">\1</p><div class="tscroll"><table>\2</table></div></div>',
                  page, flags=re.DOTALL)
    page = re.sub(r"<table\b[^>]*>(.*?)</table>", r'<div class="tscroll"><table>\1</table></div>', page,
                  flags=re.DOTALL)
    page = re.sub(r'<div class="tscroll"><div class="tscroll">(.*?)</div></div>', r'<div class="tscroll">\1</div>',
                  page, flags=re.DOTALL)
    page = re.sub(r"<colgroup>.*?</colgroup>\s*", "", page, flags=re.DOTALL)

    items = "\n".join(
        f'<li class="r r-{style}"><span class="r-id">{html.escape(label)}</span>'
        f'<span class="r-text">{html.escape(_fill(finding))}<small>{html.escape(_fill(detail))}</small></span></li>'
        for label, style, finding, detail in RESULTS)
    actions = ('<li><a href="paper.pdf">Download PDF</a></li>'
               f'<li><a href="{REPO_URL}">Code and replication</a></li>'
               f'<li><a href="{DATA_URL}">Data (Hugging Face)</a></li>'
               f'<li><a href="{REPO_URL}/blob/main/paper/thai/paper_th.pdf">ฉบับภาษาไทย (PDF)</a></li>')
    bib = ("@misc{chongmankhong2026lockdowns,\n"
           "  author = {Chongmankhong, Thanatchon},\n"
           f"  title  = {{{TITLE}: {SUBTITLE}}},\n"
           "  year   = {2026},\n"
           "  note   = {Preprint},\n"
           f"  url    = {{{PAPER_URL}}}\n"
           "}")
    cite = f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"
    colophon = ("<p>Rendered from <code>paper/template.tex</code> by <code>hesi.paper_web</code> (<code>make web</code>). "
                "Every number is filled from <code>results/</code> by the same macros as the PDF, which is "
                "authoritative; the framework diagram is a raster of the PDF's TikZ figure. Export values are "
                "simulated; the holiday calendar, OxCGRT and Google Mobility data are real.</p>")
    for marker, value in (
        ("<!--ACTIONS-->", actions),
        ("<!--RESULTS-NOTE-->", "Headline numbers from the 30-replication Monte Carlo study (simulated exports, real "
                                "calendar and COVID drivers). Green: the method works; amber: qualified; grey: it "
                                "does not, or pending."),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", cite),
        ("<!--COLOPHON-->", colophon),
    ):
        if marker not in page:
            raise RuntimeError(f"reader.html lacks {marker}")
        page = page.replace(marker, value)
    left = re.findall(r"<<[^<>]+>>|\\cite[pt]?\{", page)
    if left:
        raise RuntimeError(f"web reader has unresolved placeholders: {left[:5]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    if (paper_dir / "paper.pdf").exists():
        shutil.copyfile(paper_dir / "paper.pdf", out_dir / "paper.pdf")
    (out_dir / "figures").mkdir(exist_ok=True)
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page))):
        shutil.copyfile(paper_dir / "figures" / name, out_dir / "figures" / name)
    return out_dir / "index.html"
