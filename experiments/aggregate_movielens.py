#!/usr/bin/env python3
"""Aggregate MovieLens-100K logs across seeds."""
from __future__ import annotations
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "experiments" / "results" / "movielens"
OUT_MD = ROOT / "experiments" / "results" / "movielens_table.md"

PATTERNS = {
    # System
    "sys_p5": r"metric precision@5 = ([\d.eE+-]+)",
    "sys_p10": r"metric precision@10 = ([\d.eE+-]+)",
    "sys_p20": r"metric precision@20 = ([\d.eE+-]+)",
    "sys_ndcg10": r"metric ndcg@10 = ([\d.eE+-]+)",
    "sys_recall10": r"metric recall@10 = ([\d.eE+-]+)",
    "ild": r"metric ild = ([\d.eE+-]+)",
    "coverage": r"metric coverage = ([\d.eE+-]+)",
    "serendipity": r"metric serendipity = ([\d.eE+-]+)",
    # Baselines
    "random_p10": r"metric random_p10 = ([\d.eE+-]+)",
    "pop_p10": r"metric popularity_p10 = ([\d.eE+-]+)",
    "knn_p10": r"metric itemknn_p10 = ([\d.eE+-]+)",
    "ials_p10": r"metric ials_p10 = ([\d.eE+-]+)",
    "bpr_p10": r"metric bpr_p10 = ([\d.eE+-]+)",
    "ease_p10": r"metric ease_p10 = ([\d.eE+-]+)",
    "linucb_p10": r"metric linucb_p10 = ([\d.eE+-]+)",
    "random_ndcg10": r"metric random_ndcg10 = ([\d.eE+-]+)",
    "pop_ndcg10": r"metric popularity_ndcg10 = ([\d.eE+-]+)",
    "knn_ndcg10": r"metric itemknn_ndcg10 = ([\d.eE+-]+)",
    "ials_ndcg10": r"metric ials_ndcg10 = ([\d.eE+-]+)",
    "bpr_ndcg10": r"metric bpr_ndcg10 = ([\d.eE+-]+)",
    "ease_ndcg10": r"metric ease_ndcg10 = ([\d.eE+-]+)",
    "linucb_ndcg10": r"metric linucb_ndcg10 = ([\d.eE+-]+)",
    "pop_cov": r"metric popularity_coverage = ([\d.eE+-]+)",
    "knn_cov": r"metric itemknn_coverage = ([\d.eE+-]+)",
    "ials_cov": r"metric ials_coverage = ([\d.eE+-]+)",
    "bpr_cov": r"metric bpr_coverage = ([\d.eE+-]+)",
    "ease_cov": r"metric ease_coverage = ([\d.eE+-]+)",
    "linucb_cov": r"metric linucb_coverage = ([\d.eE+-]+)",
    # Multipliers
    "mult_random": r"metric mult_random = ([\d.eE+-]+)",
    "mult_pop": r"metric mult_popularity = ([\d.eE+-]+)",
    "mult_knn": r"metric mult_itemknn = ([\d.eE+-]+)",
    "mult_ials": r"metric mult_ials = ([\d.eE+-]+)",
    "mult_bpr": r"metric mult_bpr = ([\d.eE+-]+)",
    "mult_ease": r"metric mult_ease = ([\d.eE+-]+)",
    "mult_linucb": r"metric mult_linucb = ([\d.eE+-]+)",
    # Wilcoxon p-values + win-rates
    "wp_random": r"metric wilcoxon_random_pvalue = ([\d.eE+-]+)",
    "wp_pop": r"metric wilcoxon_popularity_pvalue = ([\d.eE+-]+)",
    "wp_knn": r"metric wilcoxon_itemknn_pvalue = ([\d.eE+-]+)",
    "wp_ials": r"metric wilcoxon_ials_pvalue = ([\d.eE+-]+)",
    "wp_bpr": r"metric wilcoxon_bpr_pvalue = ([\d.eE+-]+)",
    "wp_ease": r"metric wilcoxon_ease_pvalue = ([\d.eE+-]+)",
    "wp_linucb": r"metric wilcoxon_linucb_pvalue = ([\d.eE+-]+)",
    "ww_random": r"metric wilcoxon_random_winrate = ([\d.eE+-]+)",
    "ww_pop": r"metric wilcoxon_popularity_winrate = ([\d.eE+-]+)",
    "ww_knn": r"metric wilcoxon_itemknn_winrate = ([\d.eE+-]+)",
    "ww_ials": r"metric wilcoxon_ials_winrate = ([\d.eE+-]+)",
    "ww_bpr": r"metric wilcoxon_bpr_winrate = ([\d.eE+-]+)",
    "ww_ease": r"metric wilcoxon_ease_winrate = ([\d.eE+-]+)",
    "ww_linucb": r"metric wilcoxon_linucb_winrate = ([\d.eE+-]+)",
}


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for k, pat in PATTERNS.items():
        m = re.findall(pat, text)
        if m:
            try:
                out[k] = float(m[-1])
            except ValueError:
                pass
    return out


def stat(vals):
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def fmt_pct(s, scale=100, d=2):
    if s["n"] == 0:
        return "—"
    if s["n"] == 1:
        return f"{s['mean']*scale:.{d}f}%"
    return f"{s['mean']*scale:.{d}f}% ± {s['std']*scale:.{d}f}%"


def fmt_x(s):
    if s["n"] == 0:
        return "—"
    if s["n"] == 1:
        return f"{s['mean']:.2f}×"
    return f"{s['mean']:.2f}× ± {s['std']:.2f}"


def main():
    runs = []
    for log in sorted(LOG_DIR.glob("seed*.log")):
        d = parse_log(log)
        if d:
            runs.append(d)
    if not runs:
        print("No MovieLens logs found")
        return 1
    print(f"Loaded {len(runs)} runs")

    agg = {k: stat([r.get(k) for r in runs if r.get(k) is not None]) for k in PATTERNS}

    out = []
    out.append("# Resultados — MovieLens-100K")
    out.append("")
    out.append(f"Média sobre {len(runs)} seeds.")
    out.append("")
    out.append("## Tabela 8 (MovieLens) — Sistema vs baselines")
    out.append("")
    out.append("| Modelo | P@10 | NDCG@10 | Coverage | Multiplicador (vs Sistema) |")
    out.append("|---|---|---|---|---|")
    out.append(f"| **Sistema híbrido** | **{fmt_pct(agg['sys_p10'])}** | **{fmt_pct(agg['sys_ndcg10'])}** | **{fmt_pct(agg['coverage'])}** | — |")
    for bname, label in [
        ("random", "Aleatório"),
        ("pop", "Popularidade"),
        ("knn", "Item-kNN"),
        ("ials", "iALS (tuned)"),
        ("bpr", "BPR (tuned)"),
        ("ease", "EASE-R"),
        ("linucb", "LinUCB"),
    ]:
        p10 = agg[f"{bname}_p10"]
        ndcg = agg[f"{bname}_ndcg10"]
        cov = agg.get(f"{bname}_cov", {"n": 0, "mean": 0, "std": 0})
        mult = agg.get(f"mult_{bname}", {"n": 0, "mean": 0, "std": 0})
        out.append(f"| {label} | {fmt_pct(p10)} | {fmt_pct(ndcg)} | {fmt_pct(cov)} | {fmt_x(mult)} |")
    out.append("")

    out.append("## Tabela 9 (MovieLens) — Wilcoxon signed-rank (P@10)")
    out.append("")
    out.append("| Baseline | p-value (média) | Win-rate | Significativo (p<0.05)? |")
    out.append("|---|---|---|---|")
    for bname, label in [
        ("random", "Aleatório"),
        ("pop", "Popularidade"),
        ("knn", "Item-kNN"),
        ("ials", "iALS (tuned)"),
        ("bpr", "BPR (tuned)"),
        ("ease", "EASE-R"),
        ("linucb", "LinUCB"),
    ]:
        p = agg.get(f"wp_{bname}", {"n": 0, "mean": 1, "std": 0})
        wr = agg.get(f"ww_{bname}", {"n": 0, "mean": 0, "std": 0})
        sig = "✓" if p["mean"] < 0.05 else "✗"
        out.append(f"| {label} | {p['mean']:.4f} | {fmt_pct(wr)} | {sig} |")
    out.append("")

    OUT_MD.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    raise SystemExit(main())
