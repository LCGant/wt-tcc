#!/usr/bin/env python3
"""Generate Figure 3 with system + baselines and error bars from summary.json."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "results" / "summary.json"
OUT_PNG = ROOT / "experiments" / "results" / "figure3_corrected.png"
OUT_PNG_BASELINES = ROOT / "experiments" / "results" / "figure_baselines.png"


def main():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    by_volume = data["by_volume"]
    volumes = sorted([int(v) for v in by_volume.keys()])

    def series(metric_key: str, scale: float = 1.0):
        means, stds = [], []
        for v in volumes:
            s = by_volume[str(v)].get(metric_key, {"mean": 0, "std": 0})
            means.append(s["mean"] * scale)
            stds.append(s["std"] * scale)
        return means, stds

    # ── Figure 3 (original): convergence of system metrics ──
    p10_m, p10_s = series("sys_p10", 100)
    ndcg_m, ndcg_s = series("sys_ndcg10", 100)
    cov_m, cov_s = series("coverage", 100)
    ocr_m, ocr_s = series("ocr", 100)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    ax.errorbar(volumes, p10_m, yerr=p10_s, marker="o", linestyle="-",
                label="Precision@10", capsize=4, linewidth=1.6, markersize=7)
    ax.errorbar(volumes, ndcg_m, yerr=ndcg_s, marker="s", linestyle="-",
                label="NDCG@10", capsize=4, linewidth=1.6, markersize=7)
    ax.errorbar(volumes, cov_m, yerr=cov_s, marker="^", linestyle="-",
                label="Coverage", capsize=4, linewidth=1.6, markersize=7)
    ax.errorbar(volumes, ocr_m, yerr=ocr_s, marker="D", linestyle="-",
                label="OBC", capsize=4, linewidth=1.6, markersize=7)
    ax.set_xlabel("Número de interações por usuário", fontsize=11)
    ax.set_ylabel("Valor da métrica (%)", fontsize=11)
    ax.set_xticks(volumes)
    ax.set_xticklabels(volumes)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="center right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_PNG}")

    # ── Figure baselines: P@10 of system vs all baselines along volume ──
    sys_p10_m, sys_p10_s = series("sys_p10", 100)
    pop_p10_m, pop_p10_s = series("pop_p10", 100)
    knn_p10_m, knn_p10_s = series("knn_p10", 100)
    ials_p10_m, ials_p10_s = series("ials_p10", 100)
    bpr_p10_m, bpr_p10_s = series("bpr_p10", 100)
    ease_p10_m, ease_p10_s = series("ease_p10", 100)
    rand_p10_m, rand_p10_s = series("random_p10", 100)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    ax.errorbar(volumes, sys_p10_m, yerr=sys_p10_s, marker="o", linestyle="-",
                label="Sistema híbrido", capsize=4, linewidth=2.2, markersize=8, color="C0")
    ax.errorbar(volumes, ials_p10_m, yerr=ials_p10_s, marker="s", linestyle="--",
                label="iALS (tuned)", capsize=4, linewidth=1.4, markersize=6, color="C1")
    ax.errorbar(volumes, bpr_p10_m, yerr=bpr_p10_s, marker="^", linestyle="--",
                label="BPR (tuned)", capsize=4, linewidth=1.4, markersize=6, color="C2")
    ax.errorbar(volumes, knn_p10_m, yerr=knn_p10_s, marker="D", linestyle="--",
                label="Item-kNN", capsize=4, linewidth=1.4, markersize=6, color="C3")
    ax.errorbar(volumes, ease_p10_m, yerr=ease_p10_s, marker="P", linestyle="--",
                label="EASE-R", capsize=4, linewidth=1.4, markersize=6, color="C5")
    ax.errorbar(volumes, pop_p10_m, yerr=pop_p10_s, marker="v", linestyle=":",
                label="Popularidade", capsize=4, linewidth=1.0, markersize=6, color="C4")
    ax.errorbar(volumes, rand_p10_m, yerr=rand_p10_s, marker="x", linestyle=":",
                label="Aleatório", capsize=4, linewidth=1.0, markersize=6, color="gray")
    ax.set_xlabel("Número de interações por usuário", fontsize=11)
    ax.set_ylabel("Precision@10 (%)", fontsize=11)
    ax.set_xticks(volumes)
    ax.set_xticklabels(volumes)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.95, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_PNG_BASELINES, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_PNG_BASELINES}")


if __name__ == "__main__":
    main()
