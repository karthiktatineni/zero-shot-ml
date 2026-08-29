"""Render the current repaired-checkpoint benchmark table as a run artifact."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_DIR = Path("runs/main_run")
RESULTS_PATH = RUN_DIR / "evaluation_results.md"
OUTPUT_PATH = RUN_DIR / "evaluation_benchmarks.png"


def parse_markdown_table(path: Path) -> tuple[list[str], list[tuple[str, list[float]]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("| Dataset"))
    headers = [cell.strip() for cell in lines[header_index].strip("|").split("|")]
    rows = []

    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append((cells[0], [float(value) for value in cells[1:]]))

    return headers, rows


def main() -> None:
    headers, rows = parse_markdown_table(RESULTS_PATH)
    datasets = [dataset for dataset, _ in rows]
    scores = np.asarray([values for _, values in rows])
    model_names = headers[1:]
    colors = ["#2ca02c", "#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]

    positions = np.arange(len(datasets))
    width = 0.16
    figure, axis = plt.subplots(figsize=(14, 7.5))

    for index, model_name in enumerate(model_names):
        offset = (index - (len(model_names) - 1) / 2) * width
        axis.bar(
            positions + offset,
            scores[:, index],
            width,
            label=model_name,
            color=colors[index],
        )

    axis.set_title("Repaired 100k PFN Checkpoint vs Classical Baselines")
    axis.set_ylabel("Held-out accuracy")
    axis.set_xticks(positions, datasets)
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    figure.text(
        0.5,
        0.01,
        "PFN uses ordinal categorical encoding; baselines use one-hot encoding. "
        "Digits exceeds the PFN 20-feature capacity.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
