"""Demo of the Visualize feature with sample data (no server needed).

    cd backend && python ../scripts/visualize_demo.py [out_dir]

Prints a one-line summary of every chart view, then writes:
  visualize_demo.json          all chart payloads (small JSON, no rows)
  visualize_demo_scorecard.pdf the Trust Score scorecard with the charts embedded
The same data powers the Visualize tab in the UI (Tabular, Relational, Documents, Assistant).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from sdp import visualize as viz  # noqa: E402
from sdp.datasets import make_customers  # noqa: E402
from sdp.scoring import build_trust_report, trust_report_pdf  # noqa: E402
from sdp.tabular import TabularGenerator  # noqa: E402

VIEWS = [("customers", "distribution", {"column": "income"}), ("customers", "categorical", {"column": "plan"}), ("customers", "correlation", {}),
         ("students", "tstr", {}), ("customers", "privacy_distance", {}), ("shop_full", "cardinality", {}),
         ("bank_customers", "locale_validity", {}), ("documents-demo", "batch_summary", {})]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict] = {}
    for dataset, typ, params in VIEWS:
        p = viz.payload(viz.resolve(dataset), typ, **params)
        payloads[f"{dataset}:{typ}"] = p
        print(f"{typ:17} {dataset:15} {len(json.dumps(p)):6,} bytes  {p['caption'][:100]}")
    (out / "visualize_demo.json").write_text(json.dumps(payloads, indent=1), encoding="utf-8")

    real = make_customers(2000, seed=0)
    report = build_trust_report(real, TabularGenerator().fit(real).sample(1000, seed=1), seed=1, title="Customer table (demo data)")
    visuals = {p["type"]: p for k, p in payloads.items() if k.startswith(("customers:", "shop_full:", "bank_customers:", "documents-demo:"))}
    visuals["tstr"] = viz.payload(viz.resolve("customers"), "tstr")
    (out / "visualize_demo_scorecard.pdf").write_bytes(trust_report_pdf(report, visuals))
    print(f"\nTrust Score {report['trust_score']:.0f}/100 ({report['label']}). Wrote {out / 'visualize_demo.json'} and {out / 'visualize_demo_scorecard.pdf'}")


if __name__ == "__main__":
    main()
