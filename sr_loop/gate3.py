"""Gate 3 验收 (程序断言): 单代归档 gen_N.json 完整且全程无人工介入、无异常阻断。

判据: 文件存在; 每个子代都有终态 status (无 None); 没有 ERROR 状态 (单个候选的 ERROR 视为异常阻断);
变异统计存在 (LLM 通道或随机补齐都可, 但要如实记录); 序列项含本代/历史最佳; Pareto 前沿非空。
用法: python -m sr_loop.gate3 --archive runs/evo --gen 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def check(archive: str, gen: int) -> dict:
    p = os.path.join(archive, f"gen_{gen}.json")
    rep = {"gate": 3, "gen": gen, "file": p, "pass": False, "checks": {}}
    if not os.path.exists(p):
        rep["checks"]["exists"] = False
        return rep
    g = json.load(open(p, encoding="utf-8"))
    ch = g.get("children", [])
    rep["checks"]["exists"] = True
    rep["checks"]["n_children"] = len(ch)
    rep["checks"]["all_have_status"] = all(c.get("status") for c in ch) and len(ch) > 0
    rep["checks"]["n_error"] = sum(1 for c in ch if c.get("status") == "ERROR")
    rep["checks"]["statuses"] = {s: sum(1 for c in ch if c.get("status") == s) for s in sorted({c.get("status") for c in ch}, key=str)}
    rep["checks"]["mutation_recorded"] = gen == 0 or (g.get("llm") is not None and "provider" in g["llm"])
    rep["checks"]["llm"] = g.get("llm")
    rep["checks"]["sources"] = {s: sum(1 for c in ch if c.get("source") == s) for s in sorted({c.get("source") for c in ch}, key=str)}
    rep["checks"]["front_nonempty"] = len(g.get("front", [])) > 0
    rep["checks"]["series_entry"] = g.get("series_entry")
    rep["pass"] = (rep["checks"]["all_have_status"] and rep["checks"]["n_error"] == 0 and rep["checks"]["mutation_recorded"]
                   and rep["checks"]["front_nonempty"] and rep["checks"]["series_entry"] is not None)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="runs/evo")
    ap.add_argument("--gen", type=int, default=1)
    a = ap.parse_args()
    rep = check(a.archive, a.gen)
    json.dump(rep, open(os.path.join(a.archive, f"gate3_report_gen{a.gen}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep["checks"], ensure_ascii=False, indent=1))
    print("GATE3_PASS" if rep["pass"] else "GATE3_FAIL")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
