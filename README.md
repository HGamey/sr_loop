# sr_loop — 端侧超分网络自主进化环 (SR_LOOP_SPEC v1.1)

Mac 侧的进化环代码。物理能力 (真机延迟标尺 `bench`、游戏画面采集 `capture`) 全部由
独立 CLI `phonefarm` 提供; 本仓库只做基因组、模型生成、TFLite 导出、短训、Pareto 归档
与代际推进。规格见 `../phonefarm/docs/SPEC_SR_LOOP.md`。

```bash
# 环境 (Python 3.12 + TensorFlow 2.20, uv 建 venv)
uv venv --python 3.12 ~/.venvs/srloop && uv pip install --python ~/.venvs/srloop/bin/python -r requirements.txt

# Gate 0: 生成 Gen 0 (ESPCN, pixelshuffle / bilinear_conv 两种上采样) 并导出 TFLite
~/.venvs/srloop/bin/python -m sr_loop.gen0 --out models/gen0

# Gate 0: 真机物理标尺 (phonefarm 独立 CLI)
(cd ../phonefarm && ./phonefarm bench --serial <ID> --model ../sr_loop/models/gen0/gen0-espcn-ps.tflite --runs 3 --json)
```
