#!/usr/bin/env python3
"""Parse ablation logs (one per seed) and produce a Markdown table.

Reads .docs_sec/experiments/ablation_logs/seed*.log and outputs ablation_table.md.
"""
from __future__ import annotations
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "experiments" / "results" / "ablation"
OUT_MD = ROOT / "experiments" / "results" / "ablation_table.md"

VARIANTS = ["full", "no_thompson", "no_recent", "no_classifier", "no_exploratory"]
METRICS = ["precision@5", "precision@10", "ndcg@10", "ndcg@20"]


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    """Return {variant: {metric: value}} from one log file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
    pattern = re.compile(r"metric ablation_(\w+)_(precision@\d+|ndcg@\d+|recall@\d+) = ([\d.eE+-]+)")
    for m in pattern.finditer(text):
        variant, metric, value = m.group(1), m.group(2), m.group(3)
        if variant in out:
            try:
                out[variant][metric] = float(value)
            except ValueError:
                pass
    return out


def main():
    if not LOG_DIR.exists():
        print(f"No ablation logs at {LOG_DIR}")
        return 1

    all_runs: list[dict] = []
    for log in sorted(LOG_DIR.glob("seed*.log")):
        runs = parse_log(log)
        if any(runs.values()):
            all_runs.append(runs)

    if not all_runs:
        print("No data found in ablation logs")
        return 1

    # Aggregate by variant × metric
    agg: dict[str, dict[str, dict[str, float]]] = {}
    for variant in VARIANTS:
        agg[variant] = {}
        for metric in METRICS:
            values = []
            for run in all_runs:
                v = run.get(variant, {}).get(metric)
                if v is not None:
                    values.append(v)
            if values:
                agg[variant][metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            else:
                agg[variant][metric] = {"mean": 0.0, "std": 0.0, "n": 0}

    # Markdown
    out = []
    out.append("# Ablation Study — MovieLens-100K")
    out.append("")
    out.append(f"Average over {len(all_runs)} seeds.")
    out.append("")
    out.append("| Variante | P@5 | P@10 | NDCG@10 | NDCG@20 | Δ P@10 vs full |")
    out.append("|---|---|---|---|---|---|")

    full_p10 = agg.get("full", {}).get("precision@10", {"mean": 0})["mean"]

    for variant in VARIANTS:
        cells = []
        for metric in METRICS:
            s = agg[variant][metric]
            if s["n"] == 0:
                cells.append("—")
            elif s["n"] == 1:
                cells.append(f"{s['mean']*100:.2f}%")
            else:
                cells.append(f"{s['mean']*100:.2f}% ± {s['std']*100:.2f}%")

        # Delta vs full
        my_p10 = agg[variant]["precision@10"]["mean"]
        if variant == "full":
            delta = "—"
        else:
            d = (my_p10 - full_p10) * 100
            sign = "+" if d > 0 else ""
            delta = f"{sign}{d:.2f}pp"

        out.append(f"| `{variant}` | {' | '.join(cells)} | {delta} |")

    out.append("")
    out.append("**Interpretação:** Δ negativo significa que desabilitar esse componente *reduziu* P@10. "
               "Δ positivo significa que o componente estava prejudicando.")

    OUT_MD.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
