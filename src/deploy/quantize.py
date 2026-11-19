"""INT8 dynamic quantization of the exported ONNX models.

Weights go to INT8 and activations stay FP32, giving ~4x smaller files for
the browser without needing a calibration set.
"""
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize(input_path: str, output_path: str):
    in_size = Path(input_path).stat().st_size
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )
    out_size = Path(output_path).stat().st_size
    print(f"  {input_path} ({in_size / 1024:.1f} KB) -> "
          f"{output_path} ({out_size / 1024:.1f} KB, "
          f"{100 * out_size / in_size:.1f}%)")


def main():
    print("Quantizing asamyukta...")
    quantize("web/models/asamyukta.onnx", "web/models/asamyukta_int8.onnx")
    print("Quantizing samyukta...")
    quantize("web/models/samyukta.onnx", "web/models/samyukta_int8.onnx")


if __name__ == "__main__":
    main()
