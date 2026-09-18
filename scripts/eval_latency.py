"""Latency eval across the pipeline: classifier inference, LLM
time-to-first-token, and end-to-end (frame -> first interpretation token).

Caveats, stated up front because they matter for how to read eval/latency.json:
- Classifier inference is measured via onnxruntime's Python/native CPU
  execution provider, not the browser's onnxruntime-web WASM backend. This
  is a proxy, not a measurement of the actual in-browser number — WASM is
  typically slower than native CPU. True browser numbers require JS timing
  (performance.now() around session.run(), logged to console) which the
  app already positions to add trivially; that's a manual follow-up, not
  scripted here since there's no headless browser in this environment.
- End-to-end excludes the 1000ms stability debounce in app.js — that's a
  deliberate fixed UX delay (see STABLE_MS in web/app.js), not pipeline
  latency, and including it would make the system look artificially slow
  without reflecting an actual bottleneck.
- LLM latency depends on Groq's shared free-tier capacity and network
  conditions at run time; treat these as one representative sample, not a
  guaranteed SLA.
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SERVER_URL = "http://localhost:8001"
N_CLASSIFIER_TRIALS = 200
N_LLM_TRIALS = 10
OUT_PATH = Path("eval/latency.json")

TEST_CASES = [
    ("Pathaka", "arm raised above head"),
    ("Alapadmam", None),
    ("Mushti", "hand near chest"),
    ("Anjali", "hand near face"),
    ("Kapotham", None),
]


def percentile_stats(samples_ms: list[float]) -> dict:
    arr = sorted(samples_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(statistics.mean(arr)),
        "n": len(arr),
    }


def bench_classifier(onnx_path: str, feature_dim: int, n_trials: int) -> dict:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(1, feature_dim).astype(np.float32)

    # Warmup — first call pays for graph optimization / allocator setup.
    for _ in range(10):
        session.run(None, {input_name: x})

    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        times_ms.append((time.perf_counter() - t0) * 1000)
    return percentile_stats(times_ms)


async def bench_llm_ttft(n_trials: int) -> dict:
    times_ms = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(n_trials):
            mudra, context = TEST_CASES[i % len(TEST_CASES)]
            t0 = time.perf_counter()
            async with client.stream(
                "POST", f"{SERVER_URL}/interpret/stream",
                json={"mudras": [mudra], "context": context},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        times_ms.append((time.perf_counter() - t0) * 1000)
                        break
            await asyncio.sleep(0.3)
    return percentile_stats(times_ms)


async def main():
    print("Benchmarking asamyukta classifier (ONNX CPU, INT8)...")
    asamyukta_stats = bench_classifier(
        "web/models/asamyukta_int8.onnx", 63, N_CLASSIFIER_TRIALS
    )
    print(f"  {asamyukta_stats}")

    print("Benchmarking samyukta classifier (ONNX CPU, INT8)...")
    samyukta_stats = bench_classifier(
        "web/models/samyukta_int8.onnx", 126, N_CLASSIFIER_TRIALS
    )
    print(f"  {samyukta_stats}")

    print(f"Benchmarking LLM time-to-first-token over HTTP ({N_LLM_TRIALS} trials)...")
    llm_stats = await bench_llm_ttft(N_LLM_TRIALS)
    print(f"  {llm_stats}")

    # End-to-end = classifier inference + network/LLM TTFT, excluding the
    # deliberate 1000ms UI stability debounce (see module docstring).
    end_to_end = {
        "p50_ms": asamyukta_stats["p50_ms"] + llm_stats["p50_ms"],
        "p95_ms": asamyukta_stats["p95_ms"] + llm_stats["p95_ms"],
        "mean_ms": asamyukta_stats["mean_ms"] + llm_stats["mean_ms"],
        "note": "classifier (asamyukta, ONNX CPU proxy) + LLM TTFT; excludes "
                "the 1000ms stability debounce and browser-side WASM overhead",
    }

    output = {
        "caveats": [
            "Classifier numbers use onnxruntime's native CPU execution "
            "provider as a proxy for onnxruntime-web's WASM backend in the "
            "browser; WASM is typically slower.",
            "End-to-end excludes the 1000ms stability debounce in app.js "
            "(STABLE_MS) — that's a deliberate UX choice, not pipeline cost.",
            "LLM numbers reflect Groq free-tier capacity at run time, not a SLA.",
        ],
        "classifier_inference_ms": {
            "asamyukta_int8_onnx_cpu": asamyukta_stats,
            "samyukta_int8_onnx_cpu": samyukta_stats,
        },
        "llm_time_to_first_token_ms": llm_stats,
        "end_to_end_ms": end_to_end,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
