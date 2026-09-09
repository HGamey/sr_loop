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


def chat(system: str, user: str, max_tokens: int = 8000, temperature: float = 0.8, timeout_s: int = 240,
         attempts: int = 3) -> dict[str, Any]:
    """返回 {"ok", "provider", "text", "error", "wall_s"}; 全链失败 ok=False (调用方自行降级)。
    每家通道最多 attempts 次 (超时/限流/空回包都重试, 退避 5s*次数), 再切下一家; 错误逐条记录并打到 stderr。"""
    import sys
    load_secrets()
    errors = []
    for p in PROVIDERS:
        key = os.environ.get(p["key_env"])
        if not key:
            errors.append(f"{p['name']}: no key")
            continue
        body = {"model": p["model"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature, "max_tokens": max_tokens, **p["extra"]}
        for k in range(1, attempts + 1):
            req = urllib.request.Request(p["url"], data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    raw = resp.read().decode()
                data = json.loads(raw)
                msg0 = data["choices"][0]["message"]
                text = msg0.get("content") or msg0.get("reasoning_content") or ""
                if not text.strip():
                    # 空回包: 把 finish_reason 与回包头部记进错误, 便于定位是限流、截断还是字段变化
                    raise ValueError(f"empty content finish={data['choices'][0].get('finish_reason')} raw={raw[:240]!r}")
                return {"ok": True, "provider": p["name"], "text": text, "error": "; ".join(errors) or None, "wall_s": time.time() - t0}
            except Exception as e:  # noqa: BLE001 - 通道失败是常态, 重试后切下一家
                detail = getattr(e, "read", None)
                detail = detail().decode(errors="replace")[:200] if callable(detail) else ""
                msg = f"{p['name']} #{k}: {type(e).__name__}: {str(e)[:160]} {detail}".strip()
                errors.append(msg)
                print(f"[llm] {msg}", file=sys.stderr, flush=True)
                if k < attempts:
                    time.sleep(5 * k)
    return {"ok": False, "provider": None, "text": "", "error": "; ".join(errors), "wall_s": 0.0}
