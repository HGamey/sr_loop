"""Gate 1: 零样本真机秒筛。genome JSON → 预算校验 → 确定性建模 → TFLite → phonefarm bench → 可行域判定。

判定链 (全部程序产出, 任一环节失败即出局, 不上后续环节):
  BUDGET_FAIL   params > 50k 或 FLOPs > 2G (解析式, 不建图不上机)
  INVALID       基因组结构非法
  EXPORT_FAIL   建图/导出 TFLite 失败
  FALLBACK      真机日志有 CPU 回退 / delegate 失败
  UNSTABLE      标尺窗口不成立 (测量无效, 可重测)
  SLOW          全图 GPU 但 median > limit_ms
  FEASIBLE      全图 GPU 且 median <= limit_ms  → 进入短训
用法: python -m sr_loop.screen --serial <ID> --out runs/screen <genome.json ...>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from . import genome as G
from .bench_client import DEFAULT_LIMIT_MS, run_bench


def screen_one(g: dict[str, Any], serial: str, out_dir: str, limit_ms: float = DEFAULT_LIMIT_MS,
               runs: int = 3, fp16_file: bool = False) -> dict[str, Any]:
    t0 = time.time()
    rec: dict[str, Any] = {"id": g.get("id"), "fingerprint": None, "status": None, "budget": None,
                           "latency_ms": None, "bench_verdict": None, "tflite": None, "wall_s": None}
    try:
        G.validate(g)
    except G.GenomeError as e:
        rec.update(status="INVALID", error=str(e), wall_s=time.time() - t0)
        return rec
    rec["fingerprint"] = G.fingerprint(g)
    b = G.budget(g)
    rec["budget"] = b
    if not b["ok"]:
        rec.update(status="BUDGET_FAIL", wall_s=time.time() - t0)
        return rec
    tfl_path = os.path.join(out_dir, f"{g['id']}.tflite")
    try:
        from .export import export_tflite
        model = G.genome_to_model(g)
        exp = export_tflite(model, g, tfl_path, fp16=fp16_file)
        rec["tflite"] = {"path": tfl_path, "bytes": exp["bytes"], "ops": exp["ops"]}
    except Exception as e:  # noqa: BLE001 - 导出失败是正常的出局路径, 必须记录而非崩溃
        rec.update(status="EXPORT_FAIL", error=f"{type(e).__name__}: {e}", wall_s=time.time() - t0)
        return rec
    r = run_bench(serial, tfl_path, runs=runs, limit_ms=limit_ms, out_dir=os.path.join(out_dir, f"bench_{g['id']}"))
    rec["bench_verdict"] = r.verdict
    rec["latency_ms"] = r.latency_ms
    rec["bench_summary"] = r.raw.get("summary")
    rec["bench_json"] = os.path.join(out_dir, f"bench_{g['id']}.json")
    with open(rec["bench_json"], "w", encoding="utf-8") as f:
        json.dump(r.raw, f, ensure_ascii=False, indent=1)
    status = {"PASS": "FEASIBLE", "FAIL_LATENCY": "SLOW", "FAIL_FALLBACK": "FALLBACK",
              "FAIL_UNSTABLE": "UNSTABLE"}.get(r.verdict, "BENCH_ERROR")
    if status == "BENCH_ERROR":
        rec["error"] = r.raw.get("error") or r.raw.get("stderr_tail", "")[-800:]
    rec.update(status=status, wall_s=time.time() - t0)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--out", default="runs/screen")
    ap.add_argument("--limit-ms", type=float, default=DEFAULT_LIMIT_MS)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("genomes", nargs="+", help="genome JSON 文件")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = []
    for path in a.genomes:
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
        rec = screen_one(g, a.serial, a.out, limit_ms=a.limit_ms, runs=a.runs)
        results.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("id", "status", "latency_ms", "bench_verdict", "wall_s")}, ensure_ascii=False), flush=True)
        with open(os.path.join(a.out, "screen.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    feasible = [r["id"] for r in results if r["status"] == "FEASIBLE"]
    print(f"可行 {len(feasible)}/{len(results)}: {feasible}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
