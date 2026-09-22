import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import MNISTMLP


WIDTHS = [4, 8, 16, 32, 64, 128, 256]
CHECKPOINT_SEED = 42
WARMUP_ITERATIONS = 1000
MEASUREMENT_ITERATIONS = 10000
RESULT_PATH = Path("results/latency_results.csv")


def load_model(hidden_size: int) -> MNISTMLP:
    checkpoint_path = Path(
        f"checkpoints/mlp_h{hidden_size}_seed{CHECKPOINT_SEED}.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = MNISTMLP(hidden_size=hidden_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def benchmark_model(model: MNISTMLP, hidden_size: int) -> dict:
    example_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            model(example_input)

    latency_microseconds = np.empty(
        MEASUREMENT_ITERATIONS,
        dtype=np.float64,
    )

    garbage_collection_was_enabled = gc.isenabled()
    gc.disable()

    try:
        with torch.inference_mode():
            for iteration in range(MEASUREMENT_ITERATIONS):
                start_time = time.perf_counter_ns()
                model(example_input)
                end_time = time.perf_counter_ns()
                latency_microseconds[iteration] = (
                    end_time - start_time
                ) / 1000.0
    finally:
        if garbage_collection_was_enabled:
            gc.enable()

    median_latency = float(np.median(latency_microseconds))
    p95_latency = float(np.percentile(latency_microseconds, 95))
    mean_latency = float(np.mean(latency_microseconds))
    standard_deviation = float(np.std(latency_microseconds, ddof=1))

    return {
        "hidden_size": hidden_size,
        "checkpoint_seed": CHECKPOINT_SEED,
        "device": "cpu",
        "batch_size": 1,
        "cpu_threads": 1,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measurement_iterations": MEASUREMENT_ITERATIONS,
        "median_latency_us": median_latency,
        "p95_latency_us": p95_latency,
        "mean_latency_us": mean_latency,
        "latency_std_us": standard_deviation,
        "throughput_samples_per_second": 1_000_000.0 / median_latency,
    }


def main():
    torch.set_num_threads(1)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(2026)
    all_results = []

    print("Starting CPU single-sample latency benchmark")

    for hidden_size in WIDTHS:
        print(f"Benchmarking hidden_size={hidden_size}")
        model = load_model(hidden_size)
        result = benchmark_model(model=model, hidden_size=hidden_size)
        all_results.append(result)
        print(
            f"median={result['median_latency_us']:.3f} us, "
            f"p95={result['p95_latency_us']:.3f} us"
        )

    results = pd.DataFrame(all_results).sort_values("hidden_size")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)
    print(f"Results saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
