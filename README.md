# sr_loop — 端侧超分网络自主进化环 (SR_LOOP_SPEC v1.1)

算法侧的进化环代码。物理能力（真机延迟标尺 `bench`、游戏画面采集 `capture`）由独立实验底座 `phonefarm` 提供；本仓库负责网络结构生成、TFLite 导出、轻量训练、Pareto 归档与代际演进。规格见 `../phonefarm/docs/SPEC_SR_LOOP.md`。

---

> ## 📌 仓库定位与基准背景
>
> **端侧轻量神经网络是性能优化闭环里的候选改动方向之一，而非唯一主线。** 主线是 `phonefarm` 的评估 + 优化闭环 Harness。
>
> 全景路线见清单：
> 🔒 组织成员 [`HGamey/phonefarm/docs/MOBILE_GPU_OPT_ROUTES.md`](https://github.com/HGamey/phonefarm/blob/main/docs/MOBILE_GPU_OPT_ROUTES.md) ·
> 🌐 公开 [`BH3GEI/phonefarm/docs/MOBILE_GPU_OPT_ROUTES.md`](https://github.com/BH3GEI/phonefarm/blob/main/docs/MOBILE_GPU_OPT_ROUTES.md)
>
> **阅读本仓库实验结论时的客观背景**：
>
> - 本仓库门控指标为 `gpu`（仅含模型内核计算耗时，最优个体 `gen10-6` = 2.967 ms）。而在物理调用链中，包含内存张量拷贝与驱动同步的端到端开销约为 10 ms（540x960 -> 1080x1920 每帧拷贝约 31 MB）。
> - 端到端开销受输入载体与内存绑定机制影响显著，通过显存零拷贝绑定可有效削减往返开销，这也是后续转向 Shader / Compute 算子直通管线（如 `game_opt_loop`）的主要动力。
> - 实测表明，在限帧器封顶或 GPU 负载未饱和的场景下，降低渲染负载主要转换为整机功耗与发热预算的节省，而非直接提升帧率。

```bash
# 环境 (Python 3.12 + TensorFlow 2.20, uv 建 venv)
uv venv --python 3.12 ~/.venvs/srloop && uv pip install --python ~/.venvs/srloop/bin/python -r requirements.txt

# Gate 0: 生成 Gen 0 并导出 TFLite
~/.venvs/srloop/bin/python -m sr_loop.gen0 --out models/gen0

# Gate 0: 真机物理标尺验收
(cd ../phonefarm && ./phonefarm bench --serial <ID> --model ../sr_loop/models/gen0/gen0-espcn-ps.tflite --runs 3 --json)
~/.venvs/srloop/bin/python -m sr_loop.gate0 --serial <ID>

# Gate 1: 零样本秒筛
~/.venvs/srloop/bin/python -m sr_loop.mutate_random --parent models/gen0/gen0-espcn-ps.json --n 6 --out runs/mut
~/.venvs/srloop/bin/python -m sr_loop.screen --serial <ID> --out runs/screen runs/mut/*.json

# Gate 2: 采集样本与基准训练
(cd ../phonefarm && ./phonefarm capture --serial <ID> --out ../sr_loop/data/capture_A --frames 150 --no-shutdown --json)
~/.venvs/srloop/bin/python -m sr_loop.gate2 --capture data/capture_A --data data/A --steps 2000 --out runs/gate2

# Gate 3/4: 多代闭环演化与分析
~/.venvs/srloop/bin/python -m sr_loop.evolve --serial <ID> --data data/A --out runs/evo --generations 20 --children 6 --steps 2000
~/.venvs/srloop/bin/python -m sr_loop.gate3 --archive runs/evo --gen 1
~/.venvs/srloop/bin/python -m sr_loop.analyze --archive runs/evo
```

## 实验结果（真机物理环境）

- **Gate 0 物理标尺**：基线冷机调用稳定（重复测量离散度 <= 1.0%）。
- **Gate 1 秒筛机制**：随机变异候选快速筛除超预算模型与不合规结构。
- **Gate 2 数据集与短训**：验证样本集构建与快速收敛可行性。
- **Gate 3/4 代际演化**：13 代演化记录中，最优个体 PSNR 较 Bicubic 基线提升 +2.576 dB，真机物理标尺能够有效识别并否决超时候选。
