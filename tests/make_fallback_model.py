"""造一个必然触发 CPU 回退的 TFLite (FLOOR_MOD 不在 GPU Delegate 支持集), 用于在真机上验证
phonefarm bench 的 FAIL_FALLBACK 一票否决路径 (退出码 1, full_gpu=false, ops 里出现非 Delegate 行)。
用法: python tests/make_fallback_model.py models/test/fallback.tflite
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "models/test/fallback.tflite"
inp = tf.keras.Input(shape=(540, 960, 3), batch_size=1)
x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))(inp)
x = tf.keras.layers.Lambda(lambda t: tf.math.floormod(t, 0.5))(x)  # GPU delegate 不支持 -> 该节点留在 CPU
x = tf.keras.layers.Conv2D(12, 3, padding="same", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=2))(x)
x = tf.keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, 2))(x)
model = tf.keras.Model(inp, x)


@tf.function(input_signature=[tf.TensorSpec([1, 540, 960, 3], tf.float32)])
def f(t):
    return model(t)


frozen = convert_variables_to_constants_v2(f.get_concrete_function())
conv = tf.lite.TFLiteConverter.from_concrete_functions([frozen], model)
blob = conv.convert()
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "wb").write(blob)
interp = tf.lite.Interpreter(model_path=out)
print(out, len(blob), [d["op_name"] for d in interp._get_ops_details()])
