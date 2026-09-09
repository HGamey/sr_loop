"""Gate 0 验收 (程序断言, 无人工评分):

对 Gen 0 两个变体各做两次独立冷机 bench 调用 (每次 3 轮):
  - 每次 verdict == PASS: 全图 GPU (无 CPU 回退), 3 轮离散度 (Max-Min)/Median <= 5%, median <= 4.0ms
  - 两次冷机调用的 median 相差 <= 5% (冷机复测一致)
全部通过才输出 GATE0_PASS; 证据 (完整 JSON) 归档到 runs/gate0/<模型>_<次>.json 与 runs/gate0/report.json。

用法: python -m sr_loop.gate0 --serial <ID> [--models models/gen0/gen0-espcn-ps.tflite ...] [--out runs/gate0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .bench_client import run_bench

RETEST_TOLERANCE_PCT = 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--models", nargs="+", default=["models/gen0/gen0-espcn-ps.tflite", "models/gen0/gen0-espcn-bc.tflite"])
    ap.add_argument("--out", default="runs/gate0")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {"gate": 0, "serial": a.serial, "models": {}, "pass": True}
    for m in a.models:
        name = os.path.splitext(os.path.basename(m))[0]
        entry = {"calls": [], "pass": False}
        medians = []
        for k in (1, 2):
            r = run_bench(a.serial, m, runs=a.runs, out_dir=os.path.join(a.out, f"{name}_{k}"))
            with open(os.path.join(a.out, f"{name}_{k}.json"), "w", encoding="utf-8") as f:
                json.dump(r.raw, f, ensure_ascii=False, indent=1)
            s = r.raw.get("summary", {})
            entry["calls"].append({
                "call": k, "verdict": r.verdict, "exit": r.exit_code, "latency_ms": r.latency_ms,
                "dispersion_pct": r.dispersion_pct, "full_gpu": r.full_gpu,
                "gpu_lock_verified": s.get("gpu_lock_verified"), "error": r.raw.get("error"),
            })
            print(f"[gate0] {name} 第{k}次: verdict={r.verdict} latency={r.latency_ms} ms disp={r.dispersion_pct} "
                  f"full_gpu={r.full_gpu} lock_verified={s.get('gpu_lock_verified')}", flush=True)
            if r.verdict != "PASS" or r.latency_ms is None:
                print(f"[gate0] {name} 第{k}次未 PASS: {r.raw.get('error') or r.raw.get('stderr_tail', '')[-800:]}", flush=True)
                break
            medians.append(r.latency_ms)
        if len(medians) == 2:
            lo, hi = sorted(medians)
            retest_pct = (hi - lo) / ((hi + lo) / 2) * 100.0
            entry["retest_diff_pct"] = retest_pct
            entry["pass"] = retest_pct <= RETEST_TOLERANCE_PCT
            print(f"[gate0] {name} 冷机复测差 {retest_pct:.2f}% (上限 {RETEST_TOLERANCE_PCT}%) -> {'一致' if entry['pass'] else '不一致'}", flush=True)
        report["models"][name] = entry
        report["pass"] = report["pass"] and entry["pass"]
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("GATE0_PASS" if report["pass"] else "GATE0_FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
