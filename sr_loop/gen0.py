"""Gen 0: 最小 ESPCN 基因组 (pixelshuffle 与 bilinear_conv 两种上采样), 导出 TFLite。

用法: python -m sr_loop.gen0 --out models/gen0 [--input 540 960] [--c1 8 --c2 6]
"""
from __future__ import annotations

import argparse
import json
import os

from . import genome as G
from .export import export_tflite


def gen0_genomes(h: int, w: int, c1: int, c2: int, seed: int = 0):
    base = {
        "v": 1,
        "input": [h, w, 3],
        "scale": 2,
        "layers": [
            {"op": "conv", "k": 5, "c": c1, "act": "relu"},
            {"op": "conv", "k": 3, "c": c2, "act": "relu"},
        ],
        "seed": seed,
    }
    return [
        dict(base, id="gen0-espcn-ps", upsample="pixelshuffle"),
        dict(base, id="gen0-espcn-bc", upsample="bilinear_conv"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/gen0")
    ap.add_argument("--input", nargs=2, type=int, default=[540, 960], metavar=("H", "W"))
    ap.add_argument("--c1", type=int, default=8)
    ap.add_argument("--c2", type=int, default=6)
    ap.add_argument("--fp16", action="store_true", help="权重 fp16 量化 (缺省 fp32 权重)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = []
    for g in gen0_genomes(a.input[0], a.input[1], a.c1, a.c2):
        b = G.budget(g)
        model = G.genome_to_model(g)
        real_params = int(model.count_params())
        assert real_params == b["params"], (real_params, b["params"])
        tfl = export_tflite(model, g, os.path.join(a.out, g["id"] + ".tflite"), fp16=a.fp16)
        with open(os.path.join(a.out, g["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=1)
        row = {"id": g["id"], "fingerprint": G.fingerprint(g), **b, "tflite": tfl}
        report.append(row)
        print(json.dumps(row, ensure_ascii=False))
    with open(os.path.join(a.out, "gen0_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
