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
~/.venvs/srloop/bin/python -m sr_loop.analyze --archive runs/evo
```

产物: `runs/gate*/report.json` 为各门证据; `runs/evo/gen_N.json` 逐代归档, `runs/evo/archive.json` 指纹去重总档,
`runs/evo/spearman_report.json` 为 Gate 4 结论。LLM 通道走 `../phonefarm/secrets.env` 的密钥 (glm-5.3-flash 优先, 文本调用)。
