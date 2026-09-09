"""画质短训 (Gate 2/3): 固定种子与步数, L1 损失, 在数据集 A 上训练一个基因组并给出 val PSNR 与相对 Bicubic 的增量。

用法: python -m sr_loop.train --genome models/gen0/gen0-espcn-ps.json --data data/A --steps 2000 --out runs/train
输出: <out>/<id>.json (psnr_val, bicubic_psnr_val, delta_db, 曲线, 步数/种子/耗时) 与 <out>/<id>.weights.h5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import genome as G


def psnr_batch(pred, hr):
    import tensorflow as tf
    pred = tf.clip_by_value(pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.square(pred - hr), axis=[1, 2, 3])
    return 10.0 * tf.math.log(1.0 / tf.maximum(mse, 1e-10)) / tf.math.log(10.0)


def train_one(g: dict[str, Any], data: str, steps: int, batch: int, lr: float, seed: int, out: str,
              eval_every: int = 500) -> dict[str, Any]:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    t0 = time.time()
    G.validate(g)
    tf.keras.utils.set_random_seed(seed)
    tr_lr = np.load(os.path.join(data, "train_lr.npy"))
    tr_hr = np.load(os.path.join(data, "train_hr.npy"))
    va_lr = np.load(os.path.join(data, "val_lr.npy"))
    va_hr = np.load(os.path.join(data, "val_hr.npy"))
    baseline = json.load(open(os.path.join(data, "baseline.json"), encoding="utf-8"))
    model = G.genome_to_model(g, input_hw=None)  # 动态空间尺寸, 训练/评估用
    opt = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.CosineDecay(lr, steps, alpha=0.05))
    rng = np.random.default_rng(seed)
    n = len(tr_lr)

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            p = model(x, training=True)
            loss = tf.reduce_mean(tf.abs(p - y))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def evaluate():
        vals = []
        for i in range(0, len(va_lr), 32):
            x = va_lr[i:i + 32].astype(np.float32) / 255.0
            y = va_hr[i:i + 32].astype(np.float32) / 255.0
            vals.append(psnr_batch(model(x, training=False), y).numpy())
        return float(np.concatenate(vals).mean())

    curve = []
    loss_acc = []
    for it in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        x = tr_lr[idx].astype(np.float32) / 255.0
        y = tr_hr[idx].astype(np.float32) / 255.0
        # 增广: 随机翻转/转置 (对 LR/HR 同步), 与种子绑定
        k = int(rng.integers(0, 4))
        if k & 1:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        if k & 2:
            x, y = x[:, ::-1], y[:, ::-1]
        loss_acc.append(float(step(tf.constant(x), tf.constant(y))))
        if it % eval_every == 0 or it == steps:
            p = evaluate()
            curve.append({"step": it, "loss": float(np.mean(loss_acc)), "psnr_val": p})
            loss_acc = []
            print(f"[train {g['id']}] step {it}/{steps} loss {curve[-1]['loss']:.4f} psnr_val {p:.3f}", file=sys.stderr, flush=True)
    psnr_val = curve[-1]["psnr_val"]
    bic = baseline["bicubic_psnr_val_mean"]
    os.makedirs(out, exist_ok=True)
    wpath = os.path.join(out, f"{g['id']}.weights.h5")
    model.save_weights(wpath)
    rec = {"id": g["id"], "fingerprint": G.fingerprint(g), "psnr_val": psnr_val, "bicubic_psnr_val": bic,
           "delta_db": psnr_val - bic, "steps": steps, "batch": batch, "lr": lr, "seed": seed, "loss": "l1",
           "train_tiles": int(n), "val_tiles": int(len(va_lr)), "dataset_fingerprint": baseline["fingerprint"],
           "curve": curve, "weights": wpath, "wall_s": time.time() - t0, "params": int(model.count_params())}
    with open(os.path.join(out, f"{g['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--data", default="data/A")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/train")
    a = ap.parse_args()
    g = G.load(a.genome)
    rec = train_one(g, a.data, a.steps, a.batch, a.lr, a.seed, a.out)
    print(json.dumps({k: rec[k] for k in ("id", "psnr_val", "bicubic_psnr_val", "delta_db", "steps", "wall_s", "params")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
