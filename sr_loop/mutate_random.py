"""随机结构变异器 (Gate 1 的秒筛压力源, 也是 Gate 4 里 LLM 变异器的对照组)。

对一个父基因组做 1~3 次原子变异: 改通道 / 改核 / 换激活 / 加层 / 删层 / 换算子 / 换上采样。
只保证结构合法 (validate 通过), 不保证预算与延迟可行 —— 那是秒筛的职责。
用法: python -m sr_loop.mutate_random --parent models/gen0/gen0-espcn-ps.json --n 6 --seed 1 --out runs/mut
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any

from . import genome as G

KS = [1, 3, 5, 7]
CS = [4, 6, 8, 12, 16, 24, 32]


def mutate(parent: dict[str, Any], rng: random.Random, new_id: str) -> dict[str, Any]:
    g = copy.deepcopy(parent)
    g["id"] = new_id
    for _ in range(rng.randint(1, 3)):
        op = rng.choice(["channels", "kernel", "act", "add", "remove", "swap_op", "upsample", "skip"])
        layers = g["layers"]
        i = rng.randrange(len(layers))
        if op == "channels":
            layers[i]["c"] = rng.choice(CS)
        elif op == "kernel":
            layers[i]["k"] = rng.choice(KS)
        elif op == "act":
            layers[i]["act"] = rng.choice(sorted(G.ACTS))
        elif op == "add" and len(layers) < 16:
            layers.insert(i + 1, {"op": rng.choice(["conv", "dwsep"]), "k": rng.choice(KS), "c": rng.choice(CS), "act": "relu"})
        elif op == "remove" and len(layers) > 1:
            layers.pop(i)
        elif op == "swap_op":
            layers[i]["op"] = rng.choice(["conv", "dwsep", "res"])
        elif op == "upsample":
            g["upsample"] = rng.choice(sorted(G.UPSAMPLERS))
        elif op == "skip":
            g["skip"] = rng.choice(sorted(G.SKIPS))
    # res 块要求 c == 输入通道: 修正而不是丢弃, 让残差块真的有机会出现
    cin = 3
    for l in g["layers"]:
        if l["op"] == "res":
            l["c"] = cin
        cin = l["c"]
    G.validate(g)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/mut")
    a = ap.parse_args()
    with open(a.parent, encoding="utf-8") as f:
        parent = json.load(f)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    seen = {G.fingerprint(parent)}
    made = 0
    tries = 0
    while made < a.n and tries < a.n * 20:
        tries += 1
        g = mutate(parent, rng, f"{parent['id']}-m{a.seed}-{made + 1}")
        fp = G.fingerprint(g)
        if fp in seen:
            continue
        seen.add(fp)
        b = G.budget(g)
        path = os.path.join(a.out, g["id"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, indent=1)
        print(json.dumps({"id": g["id"], "path": path, "params": b["params"], "flops": b["flops"], "budget_ok": b["ok"],
                          "layers": [(l["op"], l["k"], l["c"], l["act"]) for l in g["layers"]], "upsample": g["upsample"]}, ensure_ascii=False))
        made += 1
    return 0


if __name__ == "__main__":
    main()
