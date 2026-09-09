"""LLM 结构变异器 (Gate 3): 输入上一代基因组与 Pareto 档案, 输出 N 个新基因组。

提示词只描述基因组语法、物理预算与真机反馈, 严禁出现任何现成论文/网络名 (铁律); id 由本模块统一分配。
LLM 全链失败或产出不足时, 用随机变异器补齐, 并在返回里如实标注 source=random。
"""
from __future__ import annotations

import json
import random
import re
from typing import Any

from . import genome as G
from .llm import chat
from .mutate_random import mutate as random_mutate

SYSTEM = """你是一个"结构变异算子"。对象是一种在手机 GPU 上运行的 2 倍图像放大小网络, 用 JSON 基因组描述。
你只输出 JSON 数组, 不输出解释。不得使用任何已发表网络或论文的名字, 也不得在 id 里写名字 (id 留空即可)。"""

SCHEMA = """基因组语法 (每个字段必填):
{"v":1,"id":"","input":[540,960,3],"scale":2,"layers":[...],"upsample":"pixelshuffle"|"bilinear_conv","seed":0}
layers 为 1..16 个层, 顺序执行, 每层:
  {"op":"conv","k":K,"c":C,"act":A}      普通卷积 KxK, 输出 C 通道
  {"op":"dwsep","k":K,"c":C,"act":A}     深度可分离: 逐通道 KxK 再 1x1 到 C 通道
  {"op":"res","k":K,"c":C,"act":A}       残差块: conv-act-conv 后与输入相加, C 必须等于该层输入通道数
  K 只能取 1/3/5/7; C 取 1..64 的整数; A 取 relu/relu6/tanh/linear
上采样头: pixelshuffle = 3x3 卷积到 12 通道后重排为 2 倍分辨率; bilinear_conv = 先双线性放大 2 倍再 3x3 卷积到 3 通道
物理预算 (在 540x960 输入上按解析式计算, 超出直接淘汰, 不上真机):
  参数 <= 50000; FLOPs (= 2*乘加) <= 2e9
  每层乘加 ≈ H*W*K*K*Cin*Cout (conv), H*W*(K*K*Cin + Cin*Cout) (dwsep), 2*H*W*K*K*C*C (res); H*W=518400
  bilinear_conv 头的卷积在 2 倍分辨率上算 (4*H*W*9*Cin*3), pixelshuffle 头为 H*W*9*Cin*12
真机门禁: GPU 内核时延 <= 4.0 ms (延迟由物理设备测得, 访存密集结构在低通道数下也可能很慢), 有任何 GPU 不支持的算子即淘汰。
目标: 在延迟门禁内最大化验证集 PSNR。
注意: 预算按上面的公式逐层相加后必须留 10% 余量 (FLOPs <= 1.8e9), 输出前请自行核算; 超预算的个体会被直接淘汰。"""


def _summary(rec: dict[str, Any]) -> dict[str, Any]:
    g = rec.get("genome") or {}
    return {"layers": [[l["op"], l["k"], l["c"], l["act"]] for l in g.get("layers", [])], "upsample": g.get("upsample"),
            "params": rec.get("params"), "flops": rec.get("flops"), "latency_ms": rec.get("latency_ms"),
            "psnr_val": rec.get("psnr_val"), "delta_db": rec.get("delta_db"), "status": rec.get("status"),
            "fallback_reason": rec.get("fallback_reason")}


def build_prompt(parents: list[dict[str, Any]], front: list[dict[str, Any]], recent: list[dict[str, Any]], n: int,
                 limit_ms: float) -> str:
    lines = [SCHEMA, "", f"延迟上限 {limit_ms} ms。", "", "当前 Pareto 前沿 (延迟 vs PSNR, 均为真机/程序实测):"]
    for r in front:
        lines.append(json.dumps(_summary(r), ensure_ascii=False))
    lines += ["", "上一代全部候选及其结局 (含淘汰原因, 供你避开死路):"]
    for r in recent:
        lines.append(json.dumps(_summary(r), ensure_ascii=False))
    lines += ["", "父代基因组 (以其为起点做结构变异, 每个子代与父代至少有一处结构差异):"]
    for r in parents:
        lines.append(json.dumps(r["genome"], ensure_ascii=False))
    lines += ["", f"请输出恰好 {n} 个新基因组组成的 JSON 数组。要求: 彼此结构不同; 至少一半用满延迟余量 (更宽或更深), "
                  "至少一个探索不同的上采样头或残差/深度可分离结构; 全部满足预算 (逐层核算); 不要复制父代。只输出 JSON。"]
    return "\n".join(lines)


def parse_genomes(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        arr = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [g for g in arr if isinstance(g, dict)]


def repair_budget(g: dict[str, Any], max_iter: int = 4) -> dict[str, Any] | None:
    """超预算的确定性修复: 按 sqrt(预算/FLOPs)*0.95 统一收缩各层通道 (res 层跟随其输入通道), 最多迭代 4 次。
    结构 (层数/算子/核/激活/上采样头) 原样保留, 只动宽度; 修不进预算返回 None。"""
    import copy
    import math
    g = copy.deepcopy(g)
    for _ in range(max_iter):
        b = G.budget(g)
        if b["ok"]:
            return g
        ratio = min(G.FLOPS_LIMIT / max(b["flops"], 1), G.PARAM_LIMIT / max(b["params"], 1))
        f = math.sqrt(ratio) * 0.95
        cin = 3
        for l in g["layers"]:
            if l["op"] == "res":
                l["c"] = cin
            else:
                l["c"] = max(1, int(l["c"] * f))
            cin = l["c"]
        try:
            G.validate(g)
        except G.GenomeError:
            return None
    return g if G.budget(g)["ok"] else None


def mutate_llm(parents: list[dict[str, Any]], front: list[dict[str, Any]], recent: list[dict[str, Any]], n: int,
               gen: int, known_fps: set[str], limit_ms: float = 4.0, seed: int = 0) -> dict[str, Any]:
    """返回 {"children": [genome...], "source": {id: llm|llm_repaired|random}, "llm": {...}}
    向 LLM 多要一倍候选, 程序侧按预算过滤; 超预算者做通道收缩修复 (标 llm_repaired); 仍不足才随机补齐。"""
    out: list[dict[str, Any]] = []
    source: dict[str, str] = {}
    fps = set(known_fps)
    info: dict[str, Any] = {"ok": False, "provider": None, "n_raw": 0, "n_valid": 0, "n_dup": 0, "n_invalid": 0,
                            "n_over_budget": 0, "n_repaired": 0, "error": None}
    res = chat(SYSTEM, build_prompt(parents, front, recent, 2 * n, limit_ms))
    info.update(ok=res["ok"], provider=res["provider"], error=res["error"], wall_s=res["wall_s"])
    over_budget: list[dict[str, Any]] = []
    if res["ok"]:
        raw = parse_genomes(res["text"])
        info["n_raw"] = len(raw)
        for g in raw:
            g = dict(g)
            g["v"] = 1
            g["id"] = ""
            g.setdefault("input", [540, 960, 3])
            g.setdefault("scale", 2)
            g.setdefault("seed", 0)
            g["id"] = "tmp"
            try:
                G.validate(g)
            except G.GenomeError:
                info["n_invalid"] += 1
                continue
            fp = G.fingerprint(g)
            if fp in fps:
                info["n_dup"] += 1
                continue
            if not G.budget(g)["ok"]:
                info["n_over_budget"] += 1
                over_budget.append(g)
                continue
            fps.add(fp)
            g["id"] = f"gen{gen}-{len(out) + 1}"
            out.append(g)
            source[g["id"]] = "llm"
            info["n_valid"] += 1
            if len(out) >= n:
                break
        for g in over_budget:
            if len(out) >= n:
                break
            r = repair_budget(g)
            if r is None:
                continue
            fp = G.fingerprint(r)
            if fp in fps:
                continue
            fps.add(fp)
            r["id"] = f"gen{gen}-{len(out) + 1}"
            out.append(r)
            source[r["id"]] = "llm_repaired"
            info["n_repaired"] += 1
    # 不足则随机变异补齐 (如实标注), 保证闭环不因外部通道断掉
    rng = random.Random(seed * 1000 + gen)
    tries = 0
    while len(out) < n and parents and tries < 200:
        tries += 1
        parent = rng.choice(parents)["genome"]
        try:
            g = random_mutate(parent, rng, f"gen{gen}-{len(out) + 1}")
        except G.GenomeError:
            continue
        fp = G.fingerprint(g)
        if fp in fps:
            continue
        fps.add(fp)
        out.append(g)
        source[g["id"]] = "random"
    return {"children": out, "source": source, "llm": info}
