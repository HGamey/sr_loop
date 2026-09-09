"""最廉价可用的文本 LLM 通道 (与 phonefarm.toml 的 provider 链同源, 按顺序失败切换)。
密钥来自环境变量或 ../phonefarm/secrets.env (只读 export KEY=... 行, 绝不执行)。"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

PROVIDERS = [
    {"name": "glm-53-flash", "url": "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions", "model": "glm-5.3-flash",
     "key_env": "GLM_KEY", "extra": {"thinking": {"type": "disabled"}}},
    {"name": "kimi-k27", "url": "https://api.kimi.com/coding/v1/chat/completions", "model": "kimi-k2.7-code", "key_env": "KIMI_KEY", "extra": {}},
    {"name": "openrouter-free", "url": "https://openrouter.ai/api/v1/chat/completions", "model": "openrouter/free", "key_env": "OPENROUTER_KEY", "extra": {}},
]
SECRETS = os.path.join(os.path.dirname(__file__), "..", "..", "phonefarm", "secrets.env")


def load_secrets() -> None:
    if not os.path.exists(SECRETS):
        return
    for line in open(SECRETS, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line[len("export "):] if line.startswith("export ") else line
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def chat(system: str, user: str, max_tokens: int = 4000, temperature: float = 0.8, timeout_s: int = 180) -> dict[str, Any]:
    """返回 {"ok", "provider", "text", "error", "wall_s"}; 全链失败 ok=False (调用方自行降级)"""
    load_secrets()
    errors = []
    for p in PROVIDERS:
        key = os.environ.get(p["key_env"])
        if not key:
            errors.append(f"{p['name']}: no key")
            continue
        body = {"model": p["model"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature, "max_tokens": max_tokens, **p["extra"]}
        req = urllib.request.Request(p["url"], data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            if not text or not text.strip():
                raise ValueError("empty content")
            return {"ok": True, "provider": p["name"], "text": text, "error": None, "wall_s": time.time() - t0}
        except Exception as e:  # noqa: BLE001 - 通道失败是常态, 切下一家
            errors.append(f"{p['name']}: {type(e).__name__}: {str(e)[:200]}")
    return {"ok": False, "provider": None, "text": "", "error": "; ".join(errors), "wall_s": 0.0}
