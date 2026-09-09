"""Gate 3/4: 单代全自主闭环与多代推进。

一代 = 变异 (LLM, 随机补齐) → 零样本真机秒筛 → 可行者短训 (固定种子/步数/L1) → 更新 Pareto 前沿 → gen_N.json。
门禁驱动, 无任何日历时间预设: 每个阶段完成即进入下一阶段; 单个候选任何异常都记为状态而不中断整代。
可续跑: 已有 gen_N.json 的代直接跳过。
用法: python -m sr_loop.evolve --serial <ID> --data data/A --out runs/evo --generations 20 --children 6 --steps 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from . import genome as G
from .mutate_llm import mutate_llm
from .screen import screen_one
from .train import train_one

STATUS_FEASIBLE = "FEASIBLE"


def pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """两目标: 延迟最小、PSNR 最大 (只在已短训的可行个体上算)"""
    pts = [r for r in records if r.get("status") == STATUS_FEASIBLE and r.get("psnr_val") is not None and r.get("latency_ms") is not None]
    front = []
    for r in pts:
        dominated = any((o["latency_ms"] <= r["latency_ms"] and o["psnr_val"] >= r["psnr_val"]
                         and (o["latency_ms"] < r["latency_ms"] or o["psnr_val"] > r["psnr_val"])) for o in pts)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["latency_ms"])


def load_archive(out: str) -> dict[str, Any]:
    p = os.path.join(out, "archive.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"records": {}, "series": []}


def save_archive(out: str, arc: dict[str, Any]) -> None:
    tmp = os.path.join(out, "archive.json.tmp")
    json.dump(arc, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(out, "archive.json"))


def evaluate(g: dict[str, Any], a, gen: int, source: str) -> dict[str, Any]:
    """秒筛 + (可行则) 短训, 返回归档记录; 任何异常落到 status 里"""
    rec: dict[str, Any] = {"id": g["id"], "gen": gen, "source": source, "genome": g, "fingerprint": G.fingerprint(g),
                           "status": None, "params": None, "flops": None, "latency_ms": None, "psnr_val": None, "delta_db": None}
    try:
        sc = screen_one(g, a.serial, os.path.join(a.out, f"gen{gen}", "screen"), limit_ms=a.limit_ms, runs=3)
        rec.update(status=sc["status"], latency_ms=sc.get("latency_ms"), bench_verdict=sc.get("bench_verdict"),
                   fallback_reason=(sc.get("bench_summary") or {}).get("fallback_reason"), screen_wall_s=sc.get("wall_s"))
        if sc.get("budget"):
            rec.update(params=sc["budget"]["params"], flops=sc["budget"]["flops"])
        if sc["status"] == "UNSTABLE":
            # 测量无效 (不是模型的问题): 重测一次, 仍不稳就如实留 UNSTABLE
            sc = screen_one(g, a.serial, os.path.join(a.out, f"gen{gen}", "screen_retry"), limit_ms=a.limit_ms, runs=3)
            rec.update(status=sc["status"], latency_ms=sc.get("latency_ms"), bench_verdict=sc.get("bench_verdict"), retried=True)
        if rec["status"] == STATUS_FEASIBLE:
            tr = train_one(g, a.data, a.steps, a.batch, a.lr, a.seed, os.path.join(a.out, f"gen{gen}", "train"))
            rec.update(psnr_val=tr["psnr_val"], delta_db=tr["delta_db"], bicubic_psnr_val=tr["bicubic_psnr_val"],
                       train_wall_s=tr["wall_s"], weights=tr["weights"])
    except Exception as e:  # noqa: BLE001 - 单个候选不许炸掉整代
        rec.update(status="ERROR", error=f"{type(e).__name__}: {e}")
    return rec


def run_generation(gen: int, a, arc: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    records = list(arc["records"].values())
    front = pareto_front(records)
    known = {r["fingerprint"] for r in records}
    gen_dir = os.path.join(a.out, f"gen{gen}")
    os.makedirs(gen_dir, exist_ok=True)
    if gen == 0:
        children = [G.load(p) for p in a.gen0]
        source = {g["id"]: "gen0" for g in children}
        llm_info = None
    else:
        recent = [r for r in records if r.get("gen") == gen - 1]
        # 父代 = Pareto 前沿; 前沿为空 (上一代全军覆没) 时退回历史最优可行个体
        parents = front or sorted([r for r in records if r.get("psnr_val") is not None], key=lambda r: -r["psnr_val"])[:2]
        m = mutate_llm(parents, front, recent, a.children, gen, known, limit_ms=a.limit_ms, seed=a.seed)
        children, source, llm_info = m["children"], m["source"], m["llm"]
        print(f"[gen{gen}] 变异: LLM={llm_info['provider']} ok={llm_info['ok']} raw={llm_info['n_raw']} valid={llm_info['n_valid']} "
              f"dup={llm_info['n_dup']} invalid={llm_info['n_invalid']}; 子代 {len(children)} (random 补齐 {sum(1 for s in source.values() if s == 'random')})",
              flush=True)
    evaluated = []
    for g in children:
        rec = evaluate(g, a, gen, source.get(g["id"], "?"))
        evaluated.append(rec)
        arc["records"][rec["fingerprint"]] = rec
        save_archive(a.out, arc)
        print(f"[gen{gen}] {rec['id']}: {rec['status']} lat={rec['latency_ms']} psnr={rec['psnr_val']} "
              f"delta={None if rec['delta_db'] is None else round(rec['delta_db'], 3)} src={rec['source']}", flush=True)
    records = list(arc["records"].values())
    front = pareto_front(records)
    feasible = [r for r in records if r.get("status") == STATUS_FEASIBLE and r.get("psnr_val") is not None]
    best = max(feasible, key=lambda r: r["psnr_val"]) if feasible else None
    gen_best = [r for r in evaluated if r.get("status") == STATUS_FEASIBLE and r.get("psnr_val") is not None]
    gen_best = max(gen_best, key=lambda r: r["psnr_val"]) if gen_best else None
    entry = {"gen": gen, "best_psnr_so_far": best["psnr_val"] if best else None, "best_id_so_far": best["id"] if best else None,
             "gen_best_psnr": gen_best["psnr_val"] if gen_best else None, "gen_best_id": gen_best["id"] if gen_best else None,
             "n_children": len(evaluated), "n_feasible": sum(1 for r in evaluated if r["status"] == STATUS_FEASIBLE),
             "front_size": len(front), "wall_s": time.time() - t0}
    arc["series"] = [s for s in arc["series"] if s["gen"] != gen] + [entry]
    arc["series"].sort(key=lambda s: s["gen"])
    save_archive(a.out, arc)
    gen_json = {"v": 1, "gen": gen, "llm": llm_info, "children": evaluated, "front": front, "series_entry": entry,
                "params": {"children": a.children, "steps": a.steps, "seed": a.seed, "limit_ms": a.limit_ms, "data": a.data}}
    json.dump(gen_json, open(os.path.join(a.out, f"gen_{gen}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[gen{gen}] 完成: 可行 {entry['n_feasible']}/{entry['n_children']}, 本代最佳 PSNR {entry['gen_best_psnr']}, "
          f"历史最佳 {entry['best_psnr_so_far']} ({entry['best_id_so_far']}), 前沿 {entry['front_size']} 个, {entry['wall_s']:.0f}s", flush=True)
    return gen_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--data", default="data/A")
    ap.add_argument("--out", default="runs/evo")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--children", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-ms", type=float, default=4.0)
    ap.add_argument("--gen0", nargs="+", default=["models/gen0/gen0-espcn-ps.json", "models/gen0/gen0-espcn-bc.json"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arc = load_archive(a.out)
    for gen in range(0, a.generations + 1):
        if os.path.exists(os.path.join(a.out, f"gen_{gen}.json")):
            print(f"[gen{gen}] 已归档, 跳过", flush=True)
            continue
        run_generation(gen, a, arc)
    print("EVOLVE_DONE", json.dumps(arc["series"][-1], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
