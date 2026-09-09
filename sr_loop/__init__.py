"""SR_LOOP: 端侧超分网络自主进化环 (SR_LOOP_SPEC v1.1) 的 Mac 侧代码。

phonefarm 只作为外部 CLI 提供 bench / capture 两个物理能力; 本包负责基因组、模型生成、
TFLite 导出、短训、Pareto 归档与代际推进。全部门禁由程序判定, 不含任何日历时间预设。
"""
