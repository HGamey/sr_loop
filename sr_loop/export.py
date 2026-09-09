"""Keras -> TFLite 导出 (fp32 权重, 静态 [1,H,W,3] 输入, GPU Delegate 友好)。

权重缺省不做 fp16 量化: 参数 <= 50k 时文件本就 < 200KB, 而 fp16 权重会给图里塞 DEQUANTIZE
节点; 真机 fp16 计算由 GPU Delegate 的 precision_loss_allowed 决定, 与文件精度无关。

只走 concrete function 通道 (Keras 3 + TF 2.20 的 from_keras_model 对 Lambda 层不稳定)。
"""
from __future__ import annotations

import os
from typing import Any


def export_tflite(model, genome: dict[str, Any], out_path: str, fp16: bool = False) -> dict[str, Any]:
    import tensorflow as tf

    h, w, c = genome["input"]

    @tf.function(input_signature=[tf.TensorSpec([1, h, w, c], tf.float32, name="lr")])
    def infer(x):
        return {"sr": model(x, training=False)}

    cf = infer.get_concrete_function()
    # 冻结: Keras 3 的权重是资源变量, 不冻结会导出 VAR_HANDLE/READ_VARIABLE/TRANSPOSE,
    # GPU Delegate 不认这些算子, 整图退回 CPU (2026-09-09 实测)
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen = convert_variables_to_constants_v2(cf)
    conv = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
    if fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    blob = conv.convert()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(blob)
    # 回读算子清单: 生成侧就能看见图里有什么, 真机日志只负责"落在哪个 delegate 上"
    ops = tflite_ops(out_path)
    return {"path": out_path, "bytes": len(blob), "ops": ops}


def tflite_ops(path: str) -> list[str]:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=path)
    ops = []
    for d in interp._get_ops_details():  # noqa: SLF001 - 官方无公开接口, 只读取名字
        ops.append(d["op_name"])
    return ops
