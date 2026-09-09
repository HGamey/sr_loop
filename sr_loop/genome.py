"""基因组 JSON 规范 v1 与确定性生成器 genome_to_model(genome) -> keras.Model。

规范 (全部字段显式, 缺省即非法):
{
  "v": 1,
  "id": "gen0-espcn-ps",                # 唯一标识 (归档键)
  "input": [540, 960, 3],               # LR 输入 H, W, C (NHWC, batch 固定 1)
  "scale": 2,                           # 放大倍数 r (整数)
  "layers": [                           # 主干, 顺序执行, 每层一个 op
    {"op": "conv", "k": 5, "c": 8, "act": "relu"},
    {"op": "conv", "k": 3, "c": 6, "act": "relu"},
    {"op": "dwsep", "k": 3, "c": 6, "act": "relu"},   # depthwise(k) + pointwise(1x1)
    {"op": "res",  "k": 3, "c": 6, "act": "relu"}     # conv-act-conv + 残差相加 (c 必须等于输入通道)
  ],
  "upsample": "pixelshuffle" | "bilinear_conv",
  "skip": "bilinear" | "none",           # 全局残差: 输出 = 网络输出 + 双线性放大的输入 (缺省 none, 向后兼容)
  "seed": 0                              # 权重初始化种子 (确定性)
}

上采样头:
  pixelshuffle : conv k3 -> C_out = 3*r*r (线性) -> depth_to_space(r)
  bilinear_conv: resize_bilinear(x r) -> conv k3 -> 3 (线性)

预算 (SR_LOOP_SPEC 硬约束, 由 budget() 程序判定): 参数 <= 50_000, FLOPs <= 2 GFLOPs
(FLOPs = 2 * MACs, 在 genome["input"] 的分辨率上按解析式计数, 不依赖任何 profiler)。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

PARAM_LIMIT = 50_000
FLOPS_LIMIT = 2_000_000_000
ACTS = {"relu", "relu6", "tanh", "linear"}
OPS = {"conv", "dwsep", "res"}
UPSAMPLERS = {"pixelshuffle", "bilinear_conv"}
SKIPS = {"bilinear", "none"}


class GenomeError(ValueError):
    pass


def validate(g: dict[str, Any]) -> None:
    """结构校验; 任何不合规直接抛 GenomeError (变异器产出的垃圾在这里被挡下)。"""
    if g.get("v") != 1:
        raise GenomeError("v 必须为 1")
    if not isinstance(g.get("id"), str) or not g["id"]:
        raise GenomeError("id 必须为非空字符串")
    inp = g.get("input")
    if not (isinstance(inp, list) and len(inp) == 3 and all(isinstance(x, int) and x > 0 for x in inp)):
        raise GenomeError("input 必须为 [H, W, C] 正整数")
    if inp[2] != 3:
        raise GenomeError("input 通道数固定为 3 (RGB)")
    r = g.get("scale")
    if not (isinstance(r, int) and r in (2, 3, 4)):
        raise GenomeError("scale 必须为 2/3/4")
    layers = g.get("layers")
    if not (isinstance(layers, list) and 1 <= len(layers) <= 16):
        raise GenomeError("layers 必须为 1..16 层")
    cin = 3
    for i, l in enumerate(layers):
        op = l.get("op")
        if op not in OPS:
            raise GenomeError(f"layers[{i}].op 非法: {op}")
        k, c, act = l.get("k"), l.get("c"), l.get("act")
        if not (isinstance(k, int) and k in (1, 3, 5, 7)):
            raise GenomeError(f"layers[{i}].k 必须为 1/3/5/7")
        if not (isinstance(c, int) and 1 <= c <= 64):
            raise GenomeError(f"layers[{i}].c 必须为 1..64")
        if act not in ACTS:
            raise GenomeError(f"layers[{i}].act 非法: {act}")
        if op == "res" and c != cin:
            raise GenomeError(f"layers[{i}] res 块要求 c == 输入通道 ({cin})")
        cin = c
    if g.get("upsample") not in UPSAMPLERS:
        raise GenomeError("upsample 必须为 pixelshuffle 或 bilinear_conv")
    if g.get("skip", "none") not in SKIPS:
        raise GenomeError("skip 必须为 bilinear 或 none")
    if not isinstance(g.get("seed"), int):
        raise GenomeError("seed 必须为整数")


def budget(g: dict[str, Any]) -> dict[str, Any]:
    """解析式统计参数量与 FLOPs (2*MACs), 在 genome 输入分辨率上计。纯函数, 不建图。"""
    validate(g)
    h, w, _ = g["input"]
    r = g["scale"]
    params = 0
    macs = 0
    cin = 3
    for l in g["layers"]:
        k, c = l["k"], l["c"]
        if l["op"] == "conv":
            params += k * k * cin * c + c
            macs += h * w * k * k * cin * c
        elif l["op"] == "dwsep":
            params += k * k * cin + cin + cin * c + c
            macs += h * w * (k * k * cin + cin * c)
        elif l["op"] == "res":
            params += 2 * (k * k * c * c + c)
            macs += 2 * h * w * k * k * c * c
        cin = c
    if g["upsample"] == "pixelshuffle":
        cout = 3 * r * r
        params += 3 * 3 * cin * cout + cout
        macs += h * w * 9 * cin * cout
    else:
        params += 3 * 3 * cin * 3 + 3
        macs += (h * r) * (w * r) * 9 * cin * 3
    if g.get("skip", "none") == "bilinear":
        # 双线性放大 (4 抽头) + 相加, 按输出像素计
        macs += (h * r) * (w * r) * 3 * 5
    flops = 2 * macs
    return {
        "params": params,
        "macs": macs,
        "flops": flops,
        "params_ok": params <= PARAM_LIMIT,
        "flops_ok": flops <= FLOPS_LIMIT,
        "ok": params <= PARAM_LIMIT and flops <= FLOPS_LIMIT,
    }


def canonical_json(g: dict[str, Any]) -> str:
    return json.dumps(g, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(g: dict[str, Any]) -> str:
    """基因组内容指纹 (不含 id), 用于去重: 同构基因组不重复上真机。"""
    body = {k: v for k, v in g.items() if k != "id"}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()[:16]


def genome_to_model(g: dict[str, Any], input_hw="genome"):
    """确定性生成 keras.Model (固定 seed 初始化; 同一 genome 两次调用权重逐位相同)。

    input_hw="genome": 用基因组的静态输入尺寸 (导出 TFLite / 真机秒筛用, GPU Delegate 要求静态形状);
    input_hw=None: 空间尺寸动态 (训练与任意尺寸评估用); 也可传 (H, W)。两种图的权重形状完全一致, 可互相载入。"""
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    validate(g)
    h, w, cch = g["input"]
    if input_hw is None:
        h, w = None, None
    elif input_hw != "genome":
        h, w = int(input_hw[0]), int(input_hw[1])
    r = g["scale"]
    seed = int(g["seed"])
    # 确定性: 每一层各自派生独立种子, 层序不变则权重不变
    counter = [seed]

    def init():
        counter[0] += 1
        return keras.initializers.GlorotUniform(seed=counter[0])

    def act_of(name):
        return None if name == "linear" else name

    inp = keras.Input(shape=(h, w, cch), batch_size=1 if h is not None else None, dtype="float32", name="lr")
    x = inp
    for l in g["layers"]:
        k, c, act = l["k"], l["c"], l["act"]
        if l["op"] == "conv":
            x = keras.layers.Conv2D(c, k, padding="same", activation=act_of(act), kernel_initializer=init())(x)
        elif l["op"] == "dwsep":
            x = keras.layers.DepthwiseConv2D(k, padding="same", depthwise_initializer=init())(x)
            x = keras.layers.Conv2D(c, 1, padding="same", activation=act_of(act), kernel_initializer=init())(x)
        elif l["op"] == "res":
            y = keras.layers.Conv2D(c, k, padding="same", activation=act_of(act), kernel_initializer=init())(x)
            y = keras.layers.Conv2D(c, k, padding="same", kernel_initializer=init())(y)
            x = keras.layers.Add()([x, y])
    if g["upsample"] == "pixelshuffle":
        x = keras.layers.Conv2D(3 * r * r, 3, padding="same", kernel_initializer=init())(x)
        x = keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, r), name="pixelshuffle")(x)
    else:
        if h is not None:
            x = keras.layers.Lambda(
                lambda t: tf.image.resize(t, (h * r, w * r), method="bilinear"), name="bilinear")(x)
        else:
            x = keras.layers.Lambda(
                lambda t: tf.image.resize(t, tf.shape(t)[1:3] * r, method="bilinear"), name="bilinear")(x)
        x = keras.layers.Conv2D(3, 3, padding="same", kernel_initializer=init())(x)
    if g.get("skip", "none") == "bilinear":
        # 全局残差: 网络只需学高频残差, 初始即等价于双线性放大 (短训收敛的关键, 2026-09-09 实测无跳连 2000 步低于 Bicubic 4 dB)
        if h is not None:
            up = keras.layers.Lambda(
                lambda t: tf.image.resize(t, (h * r, w * r), method="bilinear"), name="skip_bilinear")(inp)
        else:
            up = keras.layers.Lambda(
                lambda t: tf.image.resize(t, tf.shape(t)[1:3] * r, method="bilinear"), name="skip_bilinear")(inp)
        x = keras.layers.Add(name="skip_add")([x, up])
    model = keras.Model(inp, x, name=g["id"].replace("-", "_"))
    del np
    return model


def load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    validate(g)
    return g
