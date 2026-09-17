# sr_loop — 端侧超分网络自主进化环 (SR_LOOP_SPEC v1.1)

Mac 侧的进化环代码。物理能力 (真机延迟标尺 `bench`、游戏画面采集 `capture`) 全部由
独立 CLI `phonefarm` 提供; 本仓库只做基因组、模型生成、TFLite 导出、短训、Pareto 归档
与代际推进。规格见 `../phonefarm/docs/SPEC_SR_LOOP.md`。

```bash
# 环境 (Python 3.12 + TensorFlow 2.20, uv 建 venv)
uv venv --python 3.12 ~/.venvs/srloop && uv pip install --python ~/.venvs/srloop/bin/python -r requirements.txt

# Gate 0: 生成 Gen 0 (ESPCN, pixelshuffle / bilinear_conv 两种上采样) 并导出 TFLite
~/.venvs/srloop/bin/python -m sr_loop.gen0 --out models/gen0

# Gate 0: 真机物理标尺 (phonefarm 独立 CLI) 与验收 (两次冷机调用)
(cd ../phonefarm && ./phonefarm bench --serial <ID> --model ../sr_loop/models/gen0/gen0-espcn-ps.tflite --runs 3 --json)
~/.venvs/srloop/bin/python -m sr_loop.gate0 --serial <ID>

# Gate 1: 零样本秒筛 (预算 -> 导出 -> bench -> FEASIBLE/SLOW/FALLBACK/UNSTABLE)
~/.venvs/srloop/bin/python -m sr_loop.mutate_random --parent models/gen0/gen0-espcn-ps.json --n 6 --out runs/mut
~/.venvs/srloop/bin/python -m sr_loop.screen --serial <ID> --out runs/screen runs/mut/*.json

# Gate 2: 人先把原神进到大世界, 然后采集 -> 对齐切块 (HUD 排除/相位相关/Bicubic PSNR/熔断) -> 锁基线 -> Gen 0 短训
(cd ../phonefarm && ./phonefarm capture --serial <ID> --out ../sr_loop/data/capture_A --frames 150 --no-shutdown --json)
~/.venvs/srloop/bin/python -m sr_loop.gate2 --capture data/capture_A --data data/A --steps 2000 --out runs/gate2

# Gate 3/4: 单代闭环 x N 代 (LLM 变异 -> 秒筛 -> 短训 -> Pareto 归档 -> gen_N.json), 然后 Spearman
~/.venvs/srloop/bin/python -m sr_loop.evolve --serial <ID> --data data/A --out runs/evo --generations 20 --children 6 --steps 2000
~/.venvs/srloop/bin/python -m sr_loop.gate3 --archive runs/evo --gen 1   # Gate 3 验收: 单代归档完整且无人工介入/无异常阻断
~/.venvs/srloop/bin/python -m sr_loop.analyze --archive runs/evo
```

产物: `runs/gate*/report.json` 为各门证据; `runs/evo/gen_N.json` 逐代归档, `runs/evo/archive.json` 指纹去重总档,
`runs/evo/spearman_report.json` 为 Gate 4 结论。LLM 通道走 `../phonefarm/secrets.env` 的密钥 (glm-5.3-flash 优先, 文本调用)。

## 结果 (2026-09-09, NX809J)

- Gate 0: 标尺 PASS (Gen 0 ps/bc 两次冷机 1.099/1.088 与 1.682/1.674 ms, 复测差 1.0%/0.5%)
- Gate 1: 随机变异 6 个秒级判定 (5 FEASIBLE, 1 BUDGET_FAIL), FLOOR_MOD 模型真机 FAIL_FALLBACK
- Gate 2: 数据集 A 150 帧 1440/360 块, 丢弃率 0%, Bicubic 基线 37.286 dB; Gen 0 +1.376 / +1.537 dB
- Gate 3: gen_1.json 全自主 (LLM 6/6 可行, 零错误)
- Gate 4: 13 代 (用户决定提前结项), 累计最优序列 Spearman rho = 0.966, p = 7.7e-8; 逐代最优 rho = 0.747, p = 0.0033;
  历史最优 gen10-6: 2.967 ms, 39.862 dB (+2.576 dB vs Bicubic, +1.04 dB vs Gen 0 最佳); 第 10 代出现真机 SLOW 否决 (4.151 ms)
