"""Gate 2 验收 (程序断言): 数据集 A 构建 (对齐/切块/熔断) → 锁定 Bicubic 基线 → Gen 0 两变体短训 → 增量 >= 0.5 dB。

用法: python -m sr_loop.gate2 --capture data/capture_A --data data/A --steps 2000 --out runs/gate2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import genome as G
from .dataset import DEFAULT_CROP, build
from .train import train_one

MIN_DELTA_DB = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="data/capture_A")
    ap.add_argument("--data", default="data/A")
    ap.add_argument("--genomes", nargs="+", default=["models/gen0/gen0-espcn-ps.json", "models/gen0/gen0-espcn-bc.json"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--out", default="runs/gate2")
    ap.add_argument("--reuse-data", action="store_true", help="数据集已存在时不重建")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {"gate": 2, "pass": False, "dataset": None, "baseline": None, "models": {}}
    if not (a.reuse_data and os.path.exists(os.path.join(a.data, "baseline.json"))):
        rep = build(a.capture, a.data, 2, a.tile, DEFAULT_CROP, 28.0, 0.25, 0.05, 5, None)
        report["dataset"] = {k: rep[k] for k in rep if k not in ("hud_zones",)}
        if not rep["ok"]:
            print(f"[gate2] 熔断: 丢弃率 {rep['stats']['drop_rate']:.2%} > 5%", flush=True)
            json.dump(report, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("GATE2_FAIL")
            return 1
    else:
        report["dataset"] = json.load(open(os.path.join(a.data, "dataset_report.json"), encoding="utf-8"))
    baseline = json.load(open(os.path.join(a.data, "baseline.json"), encoding="utf-8"))
    report["baseline"] = baseline
    print(f"[gate2] 数据集 A: train {report['dataset'].get('train_tiles')} 块 / val {report['dataset'].get('val_tiles')} 块, "
          f"丢弃率 {report['dataset']['stats']['drop_rate']:.2%}, Bicubic 基线 val PSNR {baseline['bicubic_psnr_val_mean']:.3f} dB", flush=True)
    best = None
    for gp in a.genomes:
        g = G.load(gp)
        rec = train_one(g, a.data, a.steps, a.batch, a.lr, a.seed, os.path.join(a.out, "train"))
        report["models"][g["id"]] = {k: rec[k] for k in ("psnr_val", "bicubic_psnr_val", "delta_db", "steps", "seed", "params", "wall_s")}
        print(f"[gate2] {g['id']}: PSNR {rec['psnr_val']:.3f} dB, 相对 Bicubic {rec['delta_db']:+.3f} dB ({rec['wall_s']:.0f}s)", flush=True)
        if best is None or rec["delta_db"] > best[1]:
            best = (g["id"], rec["delta_db"])
    report["best"] = {"id": best[0], "delta_db": best[1], "min_delta_db": MIN_DELTA_DB}
    report["pass"] = best[1] >= MIN_DELTA_DB
    json.dump(report, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("GATE2_PASS" if report["pass"] else "GATE2_FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
