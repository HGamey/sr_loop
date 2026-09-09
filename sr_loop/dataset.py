"""Gate 2: 离线对齐切块 → 标准数据集 A。

输入: phonefarm capture 的目录 (manifest.jsonl + frame_*.png, 原始截屏)。
流程 (全部程序判定, 任一门槛不过即丢弃, 丢弃率 > --max-drop 熔断, 不产出数据集):
  1. 只取 manifest 里 state=explore 且横屏的帧
  2. 全帧铺 tile 网格, 与任一 HUD 排除区 (小地图/任务栏/顶栏/队伍列表/技能区/摇杆/血条/底栏, 相对坐标) 相交的块直接不要
     → "无 UI" 由构造保证; 相位相关只在画面中央区 (--crop) 上算
  3. LR = HR 的 bicubic 下采样 (PIL, 抗混叠 bicubic, 像素中心对齐约定与上采样一致)
  4. 相位相关平移校验: bicubic 上采样回 HR 尺寸后与 HR 做相位相关, 峰值偏移 (亚像素) 必须 <= --max-shift 像素,
     否则该帧丢弃 (对齐链路引入半像素偏移是 SR 数据集的经典暗坑, 这里用程序把它挡在门外)
  5. 切成 tile x tile 的 HR 块 (LR 块 = tile/scale), 逐块算 Bicubic 重建 PSNR, <= --min-psnr 的块丢弃
  6. 路线固定划分: manifest 的 segment (每 10 帧一段), segment % val_every == val_every-1 → val, 其余 train
  7. 落盘 npy (uint8) + split.json (帧清单/块数/阈值/Bicubic 基准 PSNR/指纹)
用法: python -m sr_loop.dataset --capture <dir> --out data/A [--tile 256] [--crop 0.16,0.14,0.84,0.82]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_CROP = (0.20, 0.15, 0.80, 0.80)
# HUD 固定区域 (l, t, r, b 相对坐标), 2026-09-09 按 NX809J 2688x1216 横屏实测帧标定
HUD_ZONES = [
    (0.00, 0.00, 1.00, 0.13),  # 顶栏: 小地图上沿/狩猎小组件/右上功能图标行
    (0.00, 0.00, 0.17, 0.40),  # 左上: 小地图 + 任务追踪
    (0.80, 0.13, 1.00, 0.50),  # 右侧: 队伍列表
    (0.60, 0.58, 1.00, 1.00),  # 右下: 攻击/技能/冲刺/元素爆发
    (0.00, 0.62, 0.25, 1.00),  # 左下: 摇杆/聊天
    (0.35, 0.85, 0.65, 1.00),  # 底部中央: 血条/等级
    (0.00, 0.93, 1.00, 1.00),  # 底栏: UID 等
]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return 99.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def phase_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    """相位相关: 返回 (dx, dy, 峰值响应). 亚像素用峰值三点抛物线拟合. ref/mov 为 float 灰度 2D."""
    ref = ref - ref.mean()
    mov = mov - mov.mean()
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f1 = np.fft.fft2(ref * win)
    f2 = np.fft.fft2(mov * win)
    cross = f1 * np.conj(f2)
    cross /= np.abs(cross) + 1e-9
    corr = np.real(np.fft.ifft2(cross))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    py, px = int(peak[0]), int(peak[1])

    def sub(c_m, c_0, c_p):
        d = c_m - 2 * c_0 + c_p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (c_m - c_p) / d

    dy = py + sub(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dx = px + sub(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dx), float(dy), float(corr[py, px])


def gray(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float64)


def tile_in_hud(tx: int, ty: int, tile: int, w: int, h: int) -> bool:
    """块 [tx,tx+tile)x[ty,ty+tile) 是否与任一 HUD 区相交 (纯函数)"""
    x0, y0, x1, y1 = tx / w, ty / h, (tx + tile) / w, (ty + tile) / h
    return any(x0 < r and x1 > l and y0 < b and y1 > t for (l, t, r, b) in HUD_ZONES)


def bicubic_down(hr: Image.Image, scale: int) -> Image.Image:
    return hr.resize((hr.width // scale, hr.height // scale), Image.BICUBIC)


def bicubic_up(lr: Image.Image, scale: int) -> Image.Image:
    return lr.resize((lr.width * scale, lr.height * scale), Image.BICUBIC)


def build(capture: str, out: str, scale: int, tile: int, crop: tuple[float, float, float, float],
          min_psnr: float, max_shift: float, max_drop: float, val_every: int, max_frames: int | None,
          segment_frames: int | None = None) -> dict[str, Any]:
    rows = [json.loads(l) for l in open(os.path.join(capture, "manifest.jsonl"), encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("state") == "explore" and r["w"] > r["h"]]
    if max_frames:
        rows = rows[:max_frames]
    if segment_frames:
        # 路线分段按保留帧序号重新划定 (manifest 缺省每 10 帧一段; 小样本冒烟或改粒度时用)
        for r in rows:
            r["segment"] = r["i"] // segment_frames
    if not rows:
        raise SystemExit("manifest 里没有横屏探索态帧")
    stats = {"frames_in": len(rows), "frames_align_drop": 0, "tiles_total": 0, "tiles_psnr_drop": 0,
             "tiles_align_drop": 0, "shifts": [], "frame_psnr": []}
    tiles: dict[str, dict[str, list]] = {"train": {"lr": [], "hr": [], "bic": [], "src": []}, "val": {"lr": [], "hr": [], "bic": [], "src": []}}
    frames_by_split: dict[str, list] = {"train": [], "val": []}
    g = 2 * scale
    stats["tiles_hud_excluded"] = 0
    for r in rows:
        img = Image.open(os.path.join(capture, r["file"])).convert("RGB")
        w, h = img.size
        # 全帧对齐到 2*scale, 再对齐到 tile 网格 (左上角起铺)
        fw, fh = w - w % g, h - h % g
        hr_img = img.crop((0, 0, fw, fh))
        lr_img = bicubic_down(hr_img, scale)
        up_img = bicubic_up(lr_img, scale)
        hr = np.asarray(hr_img)
        up = np.asarray(up_img)
        lr = np.asarray(lr_img)
        # 相位相关只看中央区 (避开 HUD 的静态图形)
        cl, ct, cr, cb = int(fw * crop[0]), int(fh * crop[1]), int(fw * crop[2]), int(fh * crop[3])
        dx, dy, resp = phase_shift(gray(hr[ct:cb, cl:cr]), gray(up[ct:cb, cl:cr]))
        stats["shifts"].append([dx, dy, resp])
        grid = [(tx, ty) for ty in range(0, fh - tile + 1, tile) for tx in range(0, fw - tile + 1, tile)]
        usable = [(tx, ty) for (tx, ty) in grid if not tile_in_hud(tx, ty, tile, fw, fh)]
        stats["tiles_hud_excluded"] += len(grid) - len(usable)
        stats["tiles_total"] += len(usable)
        if abs(dx) > max_shift or abs(dy) > max_shift:
            stats["frames_align_drop"] += 1
            stats["tiles_align_drop"] += len(usable)
            continue
        fp = psnr(up[ct:cb, cl:cr], hr[ct:cb, cl:cr])
        stats["frame_psnr"].append(fp)
        split = "val" if (r["segment"] % val_every) == val_every - 1 else "train"
        frames_by_split[split].append({"file": r["file"], "segment": r["segment"], "sha256": r["sha256"],
                                       "size": [fw, fh], "bicubic_psnr_center": fp, "shift": [dx, dy], "tiles": len(usable)})
        for (tx, ty) in usable:
            hr_t = hr[ty:ty + tile, tx:tx + tile]
            up_t = up[ty:ty + tile, tx:tx + tile]
            p = psnr(up_t, hr_t)
            if p <= min_psnr:
                stats["tiles_psnr_drop"] += 1
                continue
            lr_t = lr[ty // scale:(ty + tile) // scale, tx // scale:(tx + tile) // scale]
            tiles[split]["lr"].append(lr_t)
            tiles[split]["hr"].append(hr_t)
            tiles[split]["bic"].append(p)
            tiles[split]["src"].append([r["i"], tx, ty])
    dropped = stats["tiles_psnr_drop"] + stats["tiles_align_drop"]
    drop_rate = dropped / max(stats["tiles_total"], 1)
    stats["drop_rate"] = drop_rate
    stats["circuit_broken"] = drop_rate > max_drop
    report: dict[str, Any] = {"v": 1, "capture": capture, "scale": scale, "tile": tile, "crop": list(crop), "hud_zones": HUD_ZONES,
                              "min_psnr": min_psnr, "max_shift_px": max_shift, "max_drop": max_drop,
                              "val_every_segment": val_every, "segment_frames": segment_frames or "manifest", "stats": {k: v for k, v in stats.items() if k not in ("shifts", "frame_psnr")},
                              "shift_abs_max": [max(abs(s[0]) for s in stats["shifts"]), max(abs(s[1]) for s in stats["shifts"])],
                              "frame_bicubic_psnr_mean": float(np.mean(stats["frame_psnr"])) if stats["frame_psnr"] else None}
    os.makedirs(out, exist_ok=True)
    if stats["circuit_broken"]:
        report["ok"] = False
        with open(os.path.join(out, "dataset_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        return report
    for split in ("train", "val"):
        n = len(tiles[split]["hr"])
        if n == 0:
            raise SystemExit(f"{split} 没有任何块 (帧数太少或全被丢弃)")
        np.save(os.path.join(out, f"{split}_lr.npy"), np.stack(tiles[split]["lr"]).astype(np.uint8))
        np.save(os.path.join(out, f"{split}_hr.npy"), np.stack(tiles[split]["hr"]).astype(np.uint8))
        report[f"{split}_tiles"] = n
        report[f"{split}_bicubic_psnr_mean"] = float(np.mean(tiles[split]["bic"]))
        report[f"{split}_bicubic_psnr_median"] = float(np.median(tiles[split]["bic"]))
    h = hashlib.sha256()
    for split in ("train", "val"):
        for fr in frames_by_split[split]:
            h.update(fr["sha256"].encode())
    split_json = {**report, "ok": True, "frames": frames_by_split, "fingerprint": h.hexdigest()[:16],
                  "tile_src": {s: tiles[s]["src"] for s in ("train", "val")}}
    with open(os.path.join(out, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=1)
    baseline = {"scale": scale, "tile": tile, "val_tiles": report["val_tiles"], "bicubic_psnr_val_mean": report["val_bicubic_psnr_mean"],
                "bicubic_psnr_val_median": report["val_bicubic_psnr_median"], "fingerprint": split_json["fingerprint"]}
    with open(os.path.join(out, "baseline.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=1)
    report["ok"] = True
    report["fingerprint"] = split_json["fingerprint"]
    with open(os.path.join(out, "dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default="data/A")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--crop", default=",".join(str(c) for c in DEFAULT_CROP))
    ap.add_argument("--min-psnr", type=float, default=28.0)
    ap.add_argument("--max-shift", type=float, default=0.25)
    ap.add_argument("--max-drop", type=float, default=0.05)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--segment-frames", type=int, default=None, help="按帧序号重划路线分段 (缺省用 manifest 的 segment)")
    a = ap.parse_args()
    crop = tuple(float(x) for x in a.crop.split(","))
    rep = build(a.capture, a.out, a.scale, a.tile, crop, a.min_psnr, a.max_shift, a.max_drop, a.val_every, a.max_frames,
                a.segment_frames)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if not rep["ok"]:
        print(f"熔断: 丢弃率 {rep['stats']['drop_rate']:.3%} > {a.max_drop:.0%}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
