"""phonefarm bench 的 Python 客户端: 只调用独立 CLI, 解析其 --json 输出 (跑测分离铁律)。

phonefarm 二进制与仓库根由环境变量 PHONEFARM_DIR 指定 (缺省 ../phonefarm), bench 必须在该目录下运行
(benchmark_model 二进制按 tools/tflite/ 相对路径定位)。
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

PHONEFARM_DIR = os.environ.get("PHONEFARM_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "phonefarm"))
DEFAULT_LIMIT_MS = 4.0


@dataclass
class BenchResult:
    verdict: str
    feasible: bool
    full_gpu: bool
    latency_ms: float | None
    dispersion_pct: float | None
    exit_code: int
    raw: dict[str, Any]

    @property
    def ok_measurement(self) -> bool:
        """测量本身有效 (与是否可行无关): 非 ERROR 且离散度达标"""
        return self.verdict not in ("ERROR",) and bool(self.raw.get("summary", {}).get("dispersion_ok"))


def run_bench(serial: str, model_path: str, runs: int = 3, limit_ms: float = DEFAULT_LIMIT_MS,
              out_dir: str | None = None, extra: list[str] | None = None, timeout_s: int = 1800) -> BenchResult:
    """调用 `phonefarm bench --json`。返回结构化结果; CLI 用法错/设备错以 ERROR verdict 呈现, 不抛异常。"""
    model_abs = os.path.abspath(model_path)
    cmd = ["./phonefarm", "bench", "--serial", serial, "--model", model_abs, "--runs", str(runs),
           "--limit-ms", str(limit_ms), "--json"]
    if out_dir:
        cmd += ["--out", os.path.abspath(out_dir)]
    if extra:
        cmd += extra
    p = subprocess.run(cmd, cwd=PHONEFARM_DIR, capture_output=True, text=True, timeout=timeout_s)
    try:
        raw = json.loads(p.stdout) if p.stdout.strip() else {}
    except json.JSONDecodeError:
        raw = {"verdict": "ERROR", "error": f"非 JSON 输出: {p.stdout[:500]}"}
    if not raw:
        raw = {"verdict": "ERROR", "error": p.stderr[-2000:]}
    raw.setdefault("stderr_tail", p.stderr[-4000:])
    s = raw.get("summary", {})
    return BenchResult(
        verdict=raw.get("verdict", "ERROR"),
        feasible=bool(s.get("feasible", False)),
        full_gpu=bool(s.get("full_gpu", False)),
        latency_ms=s.get("latency_ms"),
        dispersion_pct=s.get("dispersion_pct"),
        exit_code=p.returncode,
        raw=raw,
    )


def unlock(serial: str) -> dict[str, Any]:
    p = subprocess.run(["./phonefarm", "bench", "--serial", serial, "--unlock", "--json"],
                       cwd=PHONEFARM_DIR, capture_output=True, text=True, timeout=120)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"verdict": "ERROR", "error": p.stderr[-1000:]}
