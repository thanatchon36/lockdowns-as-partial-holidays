"""Command line: `hesi <command>` or `python -m hesi.cli <command>`, run from the repository root.

  simulate  Monte Carlo study -> results/ (metrics, betas, lamhat, paths, setup)
  robust    robustness experiments R1 and R2 -> results/robust_*.csv
  figures   publication figures from results/ -> paper/figures/ (with the automated overlap check)
  paper     paper/template.tex -> paper/paper.tex -> paper/paper.pdf (tectonic, or pdflatex)
  thai      Thai draft paper/thai/paper_th.pdf (tectonic, or xelatex)
  web       web reader docs/paper/index.html (pandoc)
  real      run the methods on official monthly export data
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hesi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="Monte Carlo study")
    s.add_argument("--reps", type=int, default=30)
    s.add_argument("--seed0", type=int, default=0)
    s.add_argument("--out", default="results")
    r = sub.add_parser("robust", help="robustness experiments R1 and R2")
    r.add_argument("--reps", type=int, default=30)
    r.add_argument("--seed0", type=int, default=0)
    r.add_argument("--out", default="results")
    f = sub.add_parser("figures", help="publication figures")
    f.add_argument("--res", default="results")
    f.add_argument("--out", default="paper/figures")
    sub.add_parser("paper", help="build paper/paper.pdf")
    sub.add_parser("thai", help="build paper/thai/paper_th.pdf")
    sub.add_parser("web", help="build the web reader docs/paper/")
    x = sub.add_parser("real", help="official export data")
    x.add_argument("--exports", required=True, help="CSV: month (YYYY-MM), value")
    x.add_argument("--out", default="results_real")
    x.add_argument("--origins", nargs="*", default=["2019-12-31", "2020-12-31", "2022-12-31", "2023-12-31"])
    x.add_argument("--horizon", type=int, default=12)
    x.add_argument("--lam", type=float, default=1.0, help="dose multiplier on mobility-HESI")
    args = ap.parse_args(argv)

    if args.cmd == "simulate":
        from . import forecast
        forecast.cmd_simulate(args)
    elif args.cmd == "robust":
        from . import robust
        robust.main(args.seed0, args.reps, args.out)
    elif args.cmd == "figures":
        from . import figures
        problems = figures.make_all(res=args.res, out=args.out)
        print("overlap problems:", sum(len(v) for v in problems.values()))
    elif args.cmd == "paper":
        from . import paper_build
        return paper_build.build()
    elif args.cmd == "thai":
        from . import paper_build
        return paper_build.build_thai()
    elif args.cmd == "web":
        from . import paper_web
        print(paper_web.build_web())
    elif args.cmd == "real":
        from . import forecast
        forecast.cmd_real(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
