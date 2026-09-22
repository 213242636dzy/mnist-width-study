import argparse
from pathlib import Path

import pandas as pd

from train import run_experiment


RESULT_PATH = Path("results/raw_results.csv")


def save_result(result: dict) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([result])

    if RESULT_PATH.exists():
        existing_results = pd.read_csv(RESULT_PATH)
        same_experiment = (
            (existing_results["hidden_size"] == result["hidden_size"])
            & (existing_results["seed"] == result["seed"])
        )
        existing_results = existing_results.loc[~same_experiment]
        all_results = pd.concat(
            [existing_results, new_row],
            ignore_index=True,
        )
    else:
        all_results = new_row

    all_results = all_results.sort_values(
        by=["hidden_size", "seed"]
    ).reset_index(drop=True)
    all_results.to_csv(RESULT_PATH, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run an MNIST hidden-width sweep"
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    total_experiments = len(args.widths) * len(args.seeds)
    experiment_number = 0

    for hidden_size in args.widths:
        for seed in args.seeds:
            experiment_number += 1
            print("\n" + "=" * 70)
            print(
                f"Experiment {experiment_number}/{total_experiments}: "
                f"hidden_size={hidden_size}, seed={seed}"
            )
            print("=" * 70)

            result = run_experiment(
                hidden_size=hidden_size,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                device_name=args.device,
            )

            result["epochs_requested"] = args.epochs
            result["batch_size"] = args.batch_size
            result["learning_rate"] = args.learning_rate
            result["patience"] = args.patience
            result["device"] = args.device

            save_result(result)
            print(f"Result saved to {RESULT_PATH}")

    results = pd.read_csv(RESULT_PATH)
    print("\nCurrent results:")
    print(
        results[
            [
                "hidden_size",
                "seed",
                "best_epoch",
                "test_accuracy",
                "parameter_count",
                "macs",
                "training_seconds",
            ]
        ].to_string(index=False)
    )
