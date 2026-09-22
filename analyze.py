from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


RAW_RESULT_PATH = Path("results/raw_results.csv")
SUMMARY_PATH = Path("results/summary_results.csv")
FIGURE_PATH = Path("figures/width_accuracy.png")


def main():
    if not RAW_RESULT_PATH.exists():
        raise FileNotFoundError(f"Missing result file: {RAW_RESULT_PATH}")

    raw_results = pd.read_csv(RAW_RESULT_PATH)

    required_columns = {
        "hidden_size",
        "seed",
        "best_epoch",
        "test_accuracy",
        "parameter_count",
        "macs",
        "training_seconds",
    }

    missing_columns = required_columns - set(raw_results.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    summary = (
        raw_results.groupby("hidden_size")
        .agg(
            seed_count=("seed", "count"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            parameter_count=("parameter_count", "first"),
            macs=("macs", "first"),
            training_seconds_mean=("training_seconds", "mean"),
        )
        .reset_index()
        .sort_values("hidden_size")
    )

    summary["test_accuracy_std"] = summary["test_accuracy_std"].fillna(0.0)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    reference_width = int(summary["hidden_size"].max())
    reference_accuracy = float(
        summary.loc[
            summary["hidden_size"] == reference_width,
            "test_accuracy_mean",
        ].iloc[0]
    )

    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nReference: hidden_size={reference_width}, "
        f"mean test accuracy={reference_accuracy:.2f}%"
    )

    tolerance_results = []

    for tolerance in [0.5, 1.0]:
        target_accuracy = reference_accuracy - tolerance
        candidates = summary[
            summary["test_accuracy_mean"] >= target_accuracy
        ]
        minimum_width = (
            None
            if candidates.empty
            else int(candidates["hidden_size"].min())
        )
        tolerance_results.append(
            {
                "tolerance": tolerance,
                "target_accuracy": target_accuracy,
                "minimum_width": minimum_width,
            }
        )
        print(
            f"Tolerance {tolerance:.1f} pp: target >= "
            f"{target_accuracy:.2f}%, minimum tested width={minimum_width}"
        )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.errorbar(
        summary["hidden_size"],
        summary["test_accuracy_mean"],
        yerr=summary["test_accuracy_std"],
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        color="#2563EB",
        ecolor="#2563EB",
        label="Mean test accuracy +/- 1 SD (3 seeds)",
        zorder=3,
    )

    for _, row in summary.iterrows():
        axis.annotate(
            f'{row["test_accuracy_mean"]:.2f}%',
            (row["hidden_size"], row["test_accuracy_mean"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    line_colors = ["#DC2626", "#F59E0B"]
    for index, result in enumerate(tolerance_results):
        axis.axhline(
            result["target_accuracy"],
            linestyle="--",
            linewidth=1.3,
            color=line_colors[index],
            label=(
                f'Full width - {result["tolerance"]:.1f} pp '
                f'(min h={result["minimum_width"]})'
            ),
            zorder=2,
        )

    axis.set_xscale("log", base=2)
    axis.set_xticks(summary["hidden_size"].tolist())
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlabel("Number of hidden neurons (h)")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("MNIST Accuracy vs. Hidden-Layer Width")
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(loc="lower right", frameon=True)

    lower_error_bound = (
        summary["test_accuracy_mean"] - summary["test_accuracy_std"]
    ).min()
    upper_error_bound = (
        summary["test_accuracy_mean"] + summary["test_accuracy_std"]
    ).max()
    axis.set_ylim(
        max(0, lower_error_bound - 1),
        min(100, upper_error_bound + 0.5),
    )

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSummary saved to {SUMMARY_PATH}")
    print(f"Figure saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
