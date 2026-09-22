from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path("results/summary_results.csv")
LATENCY_PATH = Path("results/latency_results.csv")
COMBINED_PATH = Path("results/combined_results.csv")
FIGURE_PATH = Path("figures/accuracy_latency.png")


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing accuracy summary: {SUMMARY_PATH}")
    if not LATENCY_PATH.exists():
        raise FileNotFoundError(f"Missing latency results: {LATENCY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)
    latency = pd.read_csv(LATENCY_PATH)
    combined = summary.merge(
        latency,
        on="hidden_size",
        how="inner",
        validate="one_to_one",
    ).sort_values("hidden_size").reset_index(drop=True)

    reference_width = int(combined["hidden_size"].max())
    reference = combined.loc[
        combined["hidden_size"] == reference_width
    ].iloc[0]

    reference_accuracy = float(reference["test_accuracy_mean"])
    reference_parameters = float(reference["parameter_count"])
    reference_macs = float(reference["macs"])
    reference_latency = float(reference["median_latency_us"])

    combined["accuracy_drop_pp"] = (
        reference_accuracy - combined["test_accuracy_mean"]
    )
    combined["parameter_reduction_percent"] = (
        1.0 - combined["parameter_count"] / reference_parameters
    ) * 100.0
    combined["macs_reduction_percent"] = (
        1.0 - combined["macs"] / reference_macs
    ) * 100.0
    combined["latency_reduction_percent"] = (
        1.0 - combined["median_latency_us"] / reference_latency
    ) * 100.0
    combined["latency_speedup"] = (
        reference_latency / combined["median_latency_us"]
    )

    COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_PATH, index=False)

    print("\nAccuracy-resource-latency results:")
    print(
        combined[
            [
                "hidden_size",
                "test_accuracy_mean",
                "test_accuracy_std",
                "accuracy_drop_pp",
                "macs_reduction_percent",
                "median_latency_us",
                "p95_latency_us",
                "latency_reduction_percent",
                "latency_speedup",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.errorbar(
        combined["median_latency_us"],
        combined["test_accuracy_mean"],
        yerr=combined["test_accuracy_std"],
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=4,
        capthick=1.5,
        color="#2563EB",
        ecolor="#2563EB",
        label="Mean accuracy +/- 1 SD",
    )

    for _, row in combined.iterrows():
        axis.annotate(
            f"h={int(row['hidden_size'])}",
            (row["median_latency_us"], row["test_accuracy_mean"]),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=9,
        )

    lower_accuracy = (
        combined["test_accuracy_mean"] - combined["test_accuracy_std"]
    ).min()
    upper_accuracy = (
        combined["test_accuracy_mean"] + combined["test_accuracy_std"]
    ).max()

    axis.set_xlim(
        combined["median_latency_us"].min() - 0.2,
        combined["median_latency_us"].max() + 0.2,
    )
    axis.set_ylim(
        max(0, lower_accuracy - 1),
        min(100, upper_accuracy + 0.5),
    )
    axis.set_xlabel("Median CPU latency per sample (us)")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("MNIST Accuracy-Latency Trade-off")
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"\nCombined results saved to {COMBINED_PATH}")
    print(f"Figure saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
