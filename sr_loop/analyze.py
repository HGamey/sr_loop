"""Gate 4: 20 代演进序列的 Spearman 秩相关分析 → 单调演进结论报告 (rho, p-value)。
序列 = 每代结束时可行域 (<= limit_ms, 全图 GPU) 内的历史最高 PSNR (best_psnr_so_far) 与本代最高 PSNR (gen_best_psnr) 两条。
用法: python -m sr_loop.analyze --archive runs/evo --out runs/evo/spearman_report.json
"""
from __future__ import annotations

import argparse
import json
import sys


def spearman(xs, ys):
    from scipy.stats import spearmanr
    r = spearmanr(xs, ys)
    return float(r.statistic), float(r.pvalue)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="runs/evo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    arc = json.load(open(f"{a.archive}/archive.json", encoding="utf-8"))
    series = [s for s in arc["series"] if s.get("gen", 0) >= 1]
    gens = [s["gen"] for s in series]
    report = {"generations": len(series), "alpha": a.alpha, "series": series}
    for key in ("best_psnr_so_far", "gen_best_psnr"):
        pairs = [(g, s[key]) for g, s in zip(gens, series) if s.get(key) is not None]
        if len(pairs) >= 3:
            rho, p = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            report[key] = {"n": len(pairs), "rho": rho, "p_value": p, "significant_positive": rho > 0 and p < a.alpha,
                           "first": pairs[0][1], "last": pairs[-1][1], "gain_db": pairs[-1][1] - pairs[0][1]}
        else:
            report[key] = {"n": len(pairs), "rho": None, "p_value": None, "significant_positive": False}
    cum = report["best_psnr_so_far"]
    per = report["gen_best_psnr"]
    report["conclusion"] = ("显著正向演进" if cum.get("significant_positive") else "未见显著单调演进") + \
        f" (累计最优: rho={cum.get('rho')}, p={cum.get('p_value')}; 逐代最优: rho={per.get('rho')}, p={per.get('p_value')})"
    out = a.out or f"{a.archive}/spearman_report.json"
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("generations", "best_psnr_so_far", "gen_best_psnr", "conclusion")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
