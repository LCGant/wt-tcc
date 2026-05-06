#!/usr/bin/env python3
"""Parse all experiment logs and produce results.csv + summary.json + tables.md.

Runs entirely from logs in .docs_sec/experiments/logs/seed<S>_vol<V>.log
"""
from __future__ import annotations
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "results" / "pipeline"
LOG_DIR = EXP_DIR
OUT_DIR = ROOT / "experiments" / "results"
CSV_PATH = OUT_DIR / "results.csv"
JSON_PATH = OUT_DIR / "summary.json"
MD_PATH = OUT_DIR / "tables.md"

# Field name -> regex pattern (extracted from log lines)
PATTERNS = {
    "users": r"^    users:\s+(\d+)",
    "places": r"^    places:\s+(\d+)",
    "signals_total": r"^    signals:\s+(\d+)",
    "train_size": r"^    train split:\s+(\d+)",
    "test_size": r"^    test split:\s+(\d+)",
    "density": r"metric density = ([\d.eE+-]+)",
    "sys_p5": r"metric precision@5 = ([\d.eE+-]+)",
    "sys_p10": r"metric precision@10 = ([\d.eE+-]+)",
    "sys_p20": r"metric precision@20 = ([\d.eE+-]+)",
    "sys_ndcg5": r"metric ndcg@5 = ([\d.eE+-]+)",
    "sys_ndcg10": r"metric ndcg@10 = ([\d.eE+-]+)",
    "sys_ndcg20": r"metric ndcg@20 = ([\d.eE+-]+)",
    "sys_recall5": r"metric recall@5 = ([\d.eE+-]+)",
    "sys_recall10": r"metric recall@10 = ([\d.eE+-]+)",
    "sys_recall20": r"metric recall@20 = ([\d.eE+-]+)",
    "ild": r"^    ILD \(diversity\):\s+([\d.]+)",
    "coverage": r"^    Coverage:\s+([\d.]+)",
    "serendipity": r"^    Serendipity:\s+([\d.]+)",
    "ocr": r"^    OCR \(correction\):\s+([\d.]+)",
    "random_p10": r"metric random_p10 = ([\d.eE+-]+)",
    "pop_p10": r"metric popularity_p10 = ([\d.eE+-]+)",
    "knn_p10": r"metric itemknn_p10 = ([\d.eE+-]+)",
    "ials_p10": r"metric ials_p10 = ([\d.eE+-]+)",
    "bpr_p10": r"metric bpr_p10 = ([\d.eE+-]+)",
    "ease_p10": r"metric ease_p10 = ([\d.eE+-]+)",
    "random_ndcg10": r"metric random_ndcg10 = ([\d.eE+-]+)",
    "pop_ndcg10": r"metric popularity_ndcg10 = ([\d.eE+-]+)",
    "knn_ndcg10": r"metric itemknn_ndcg10 = ([\d.eE+-]+)",
    "ials_ndcg10": r"metric ials_ndcg10 = ([\d.eE+-]+)",
    "bpr_ndcg10": r"metric bpr_ndcg10 = ([\d.eE+-]+)",
    "ease_ndcg10": r"metric ease_ndcg10 = ([\d.eE+-]+)",
    "random_cov": r"metric random_coverage = ([\d.eE+-]+)",
    "pop_cov": r"metric popularity_coverage = ([\d.eE+-]+)",
    "knn_cov": r"metric itemknn_coverage = ([\d.eE+-]+)",
    "ials_cov": r"metric ials_coverage = ([\d.eE+-]+)",
    "bpr_cov": r"metric bpr_coverage = ([\d.eE+-]+)",
    "ease_cov": r"metric ease_coverage = ([\d.eE+-]+)",
    "mult_random": r"metric mult_random = ([\d.eE+-]+)",
    "mult_popularity": r"metric mult_popularity = ([\d.eE+-]+)",
    "mult_itemknn": r"metric mult_itemknn = ([\d.eE+-]+)",
    "mult_ials": r"metric mult_ials = ([\d.eE+-]+)",
    "mult_bpr": r"metric mult_bpr = ([\d.eE+-]+)",
    "mult_ease": r"metric mult_ease = ([\d.eE+-]+)",
    "wilcoxon_random_p": r"metric wilcoxon_random_pvalue = ([\d.eE+-]+)",
    "wilcoxon_pop_p": r"metric wilcoxon_popularity_pvalue = ([\d.eE+-]+)",
    "wilcoxon_knn_p": r"metric wilcoxon_itemknn_pvalue = ([\d.eE+-]+)",
    "wilcoxon_ials_p": r"metric wilcoxon_ials_pvalue = ([\d.eE+-]+)",
    "wilcoxon_bpr_p": r"metric wilcoxon_bpr_pvalue = ([\d.eE+-]+)",
    "wilcoxon_ease_p": r"metric wilcoxon_ease_pvalue = ([\d.eE+-]+)",
    "wilcoxon_random_winrate": r"metric wilcoxon_random_winrate = ([\d.eE+-]+)",
    "wilcoxon_pop_winrate": r"metric wilcoxon_popularity_winrate = ([\d.eE+-]+)",
    "wilcoxon_knn_winrate": r"metric wilcoxon_itemknn_winrate = ([\d.eE+-]+)",
    "wilcoxon_ials_winrate": r"metric wilcoxon_ials_winrate = ([\d.eE+-]+)",
    "wilcoxon_bpr_winrate": r"metric wilcoxon_bpr_winrate = ([\d.eE+-]+)",
    "wilcoxon_ease_winrate": r"metric wilcoxon_ease_winrate = ([\d.eE+-]+)",
}

INT_FIELDS = {"users", "places", "signals_total", "train_size", "test_size"}


def parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for field, pattern in PATTERNS.items():
        compiled = re.compile(pattern, re.MULTILINE)
        matches = compiled.findall(text)
        if not matches:
            continue
        val = matches[-1]  # take last occurrence
        try:
            out[field] = int(val) if field in INT_FIELDS else float(val)
        except (ValueError, TypeError):
            pass
    return out


def parse_filename(path: Path) -> tuple[int, int] | None:
    m = re.match(r"seed(\d+)_vol(\d+)\.log", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def collect_rows() -> list[dict]:
    rows = []
    for log in sorted(LOG_DIR.glob("seed*_vol*.log")):
        meta = parse_filename(log)
        if not meta:
            continue
        seed, volume = meta
        metrics = parse_log(log)
        if not metrics:
            continue
        row = {"seed": seed, "volume": volume, **metrics}
        rows.append(row)
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "volume"] + list(PATTERNS.keys())
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {CSV_PATH} ({len(rows)} rows)")


def stats(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not clean:
        return {"mean": 0.0, "std": 0.0, "n": 0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
        "min": min(clean),
        "max": max(clean),
    }


def aggregate(rows: list[dict]) -> dict:
    by_volume: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_volume[r["volume"]].append(r)

    summary = {
        "by_volume": {},
        "global": {},
        "n_runs": len(rows),
        "n_seeds": len({r["seed"] for r in rows}),
    }
    metrics_list = list(PATTERNS.keys())
    for vol, vol_rows in sorted(by_volume.items()):
        summary["by_volume"][str(vol)] = {m: stats([r.get(m) for r in vol_rows]) for m in metrics_list}
    summary["global"] = {m: stats([r.get(m) for r in rows]) for m in metrics_list}
    return summary


def fmt_pct(s: dict, scale: float = 100, digits: int = 2) -> str:
    if s["n"] == 0:
        return "—"
    if s["n"] == 1:
        return f"{s['mean']*scale:.{digits}f}%"
    return f"{s['mean']*scale:.{digits}f}% ± {s['std']*scale:.{digits}f}%"


def fmt_num(s: dict, digits: int = 4) -> str:
    if s["n"] == 0:
        return "—"
    if s["n"] == 1:
        return f"{s['mean']:.{digits}f}"
    return f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f}"


def fmt_x(s: dict) -> str:
    if s["n"] == 0:
        return "—"
    if s["n"] == 1:
        return f"{s['mean']:.2f}×"
    return f"{s['mean']:.2f}× ± {s['std']:.2f}"


def write_markdown(summary: dict) -> None:
    out = []
    out.append("# Resultados Agregados — Experimentos com sistema híbrido + 5 baselines")
    out.append("")
    out.append(f"**N execuções:** {summary['n_runs']} | **N seeds únicas:** {summary['n_seeds']}")
    out.append("")
    out.append("Valores reportados como `média ± desvio-padrão`.")
    out.append("")

    v80 = summary["by_volume"].get("80", {})

    # Tabela 6
    out.append("## Tabela 6 — Métricas de ranqueamento (regime warm, 80 int./usuário)")
    out.append("")
    out.append("| K | Precision@K | Recall@K | NDCG@K |")
    out.append("|---|---|---|---|")
    for k in (5, 10, 20):
        out.append(
            f"| {k} | {fmt_pct(v80.get(f'sys_p{k}', {'n':0,'mean':0,'std':0}))} | "
            f"{fmt_pct(v80.get(f'sys_recall{k}', {'n':0,'mean':0,'std':0}))} | "
            f"{fmt_pct(v80.get(f'sys_ndcg{k}', {'n':0,'mean':0,'std':0}))} |"
        )
    out.append("")

    # Tabela 7
    out.append("## Tabela 7 — Indicadores comportamentais")
    out.append("")
    out.append("| Métrica | Valor (média ± DP) |")
    out.append("|---|---|")
    out.append(f"| ILD (diversidade) | {fmt_num(v80.get('ild', {'n':0,'mean':0,'std':0}))} |")
    out.append(f"| Coverage | {fmt_pct(v80.get('coverage', {'n':0,'mean':0,'std':0}))} |")
    out.append(f"| Serendipity | {fmt_pct(v80.get('serendipity', {'n':0,'mean':0,'std':0}))} |")
    out.append(f"| OBC (correção do onboarding) | {fmt_pct(v80.get('ocr', {'n':0,'mean':0,'std':0}))} |")
    out.append("")

    # Tabela 8 — Sistema vs todos baselines
    out.append("## Tabela 8 — Sistema híbrido vs todos os baselines (regime warm)")
    out.append("")
    out.append("| Modelo | P@10 | NDCG@10 | Coverage | vs Sistema (mult.) |")
    out.append("|---|---|---|---|---|")
    sys_p10 = v80.get("sys_p10", {"n": 0, "mean": 0, "std": 0})
    sys_ndcg = v80.get("sys_ndcg10", {"n": 0, "mean": 0, "std": 0})
    sys_cov = v80.get("coverage", {"n": 0, "mean": 0, "std": 0})
    out.append(
        f"| **Sistema híbrido** | **{fmt_pct(sys_p10)}** | **{fmt_pct(sys_ndcg)}** | "
        f"**{fmt_pct(sys_cov)}** | — |"
    )
    for bname, label in [
        ("random", "Aleatório"),
        ("pop", "Popularidade"),
        ("knn", "Item-kNN"),
        ("ials", "iALS (tuned)"),
        ("bpr", "BPR (tuned)"),
        ("ease", "EASE-R"),
    ]:
        p10 = v80.get(f"{bname}_p10", {"n": 0, "mean": 0, "std": 0})
        ndcg = v80.get(f"{bname}_ndcg10", {"n": 0, "mean": 0, "std": 0})
        cov = v80.get(f"{bname}_cov", {"n": 0, "mean": 0, "std": 0})
        mult_key = f"mult_{bname.replace('pop', 'popularity').replace('knn', 'itemknn')}"
        mult = v80.get(mult_key, {"n": 0, "mean": 0, "std": 0})
        out.append(
            f"| {label} | {fmt_pct(p10)} | {fmt_pct(ndcg)} | "
            f"{fmt_pct(cov)} | {fmt_x(mult)} |"
        )
    out.append("")

    # Tabela 9 — Wilcoxon por baseline
    out.append("## Tabela 9 — Significância estatística (Wilcoxon signed-rank, P@10)")
    out.append("")
    out.append("| Baseline | p-value (média) | Win-rate (média ± DP) | Significativo (p<0.05)? |")
    out.append("|---|---|---|---|")
    for bname, label in [
        ("random", "Aleatório"),
        ("pop", "Popularidade"),
        ("knn", "Item-kNN"),
        ("ials", "iALS (tuned)"),
        ("bpr", "BPR (tuned)"),
        ("ease", "EASE-R"),
    ]:
        p_stats = v80.get(f"wilcoxon_{bname}_p", {"n": 0, "mean": 1, "std": 0})
        wr = v80.get(f"wilcoxon_{bname}_winrate", {"n": 0, "mean": 0, "std": 0})
        sig = "✓" if p_stats["mean"] < 0.05 else "✗"
        out.append(
            f"| {label} | {p_stats['mean']:.4f} | {fmt_pct(wr)} | {sig} |"
        )
    out.append("")

    # Convergence (Fig 3)
    out.append("## Dados para Figura 3 — Curva de convergência")
    out.append("")
    out.append("| Volume | P@10 | NDCG@10 | Coverage | OBC | iALS P@10 | kNN P@10 |")
    out.append("|---|---|---|---|---|---|---|")
    for vol in [2, 5, 10, 20, 40, 80]:
        v = summary["by_volume"].get(str(vol), {})
        if not v:
            continue
        out.append(
            f"| {vol} | {fmt_pct(v.get('sys_p10', {'n':0,'mean':0,'std':0}))} | "
            f"{fmt_pct(v.get('sys_ndcg10', {'n':0,'mean':0,'std':0}))} | "
            f"{fmt_pct(v.get('coverage', {'n':0,'mean':0,'std':0}))} | "
            f"{fmt_pct(v.get('ocr', {'n':0,'mean':0,'std':0}))} | "
            f"{fmt_pct(v.get('ials_p10', {'n':0,'mean':0,'std':0}))} | "
            f"{fmt_pct(v.get('knn_p10', {'n':0,'mean':0,'std':0}))} |"
        )
    out.append("")

    MD_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {MD_PATH}")


def main():
    rows = collect_rows()
    if not rows:
        print("No logs found in", LOG_DIR)
        return 1
    print(f"Loaded {len(rows)} rows from {LOG_DIR}")
    write_csv(rows)
    summary = aggregate(rows)
    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {JSON_PATH}")
    write_markdown(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
