from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path("/benchmark-eeg/5.0_version")
INPUT_DIR = REPO_ROOT / "analysis" / "result_5"
OUT_DIR = REPO_ROOT / "analysis" / "six_category_radar"

MODEL_ORDER = [
    "brainomni",
    "labram",
    "biot",
    "femba",
    "neurolm",
    "cbramod",
    "neurogpt",
    "reve",
    "eegmamba",
    "bendr",
]

MODEL_LABEL = {
    "brainomni": "BrainOmni",
    "labram": "LaBraM",
    "biot": "BIOT",
    "femba": "FEMBA",
    "neurolm": "NeuroLM",
    "cbramod": "CBraMod",
    "neurogpt": "NeuroGPT",
    "reve": "REVE",
    "eegmamba": "EEGMamba",
    "bendr": "BENDR",
}

DISPLAY_TO_MODEL = {v: k for k, v in MODEL_LABEL.items()}

CATEGORY_ORDER = ["Type-I", "Type-II", "Type-III", "Type-IV", "Type-V", "Type-VI"]
CATEGORY_LABEL = {
    "Type-I": "Signal\nReliability",
    "Type-II": "Biometrics\n& Disease",
    "Type-III": "Consciousness\n& State",
    "Type-IV": "Cognition\n& Emotion",
    "Type-V": "Naturalistic\nStimulus Decoding",
    "Type-VI": "Motor\n& Interaction",
}


def load_category_rank(input_dir: Path, split: str) -> pd.DataFrame:
    mode = "cross_subject" if split == "cross" else "within_subject"
    path = input_dir / f"linear_full_mean_accuracy_rank_by_type_{mode}.csv"
    cat_rank = pd.read_csv(path)
    if "type" not in cat_rank.columns:
        raise KeyError(f"Missing required 'type' column in {path}")

    cat_rank = cat_rank.set_index("type").reindex(CATEGORY_ORDER).dropna(how="all")
    cat_rank = cat_rank.rename(columns=DISPLAY_TO_MODEL)

    model_cols = [m for m in MODEL_ORDER if m in cat_rank.columns]
    cat_rank = cat_rank[model_cols]

    # Match linear_downsample_cross_radar.py: models are ordered by lower mean rank.
    model_order = cat_rank.mean(axis=0, skipna=True).sort_values().index.tolist()
    return cat_rank[model_order]


def plot_radar(cat_rank: pd.DataFrame, out_dir: Path, stem: str, split: str = "cross") -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
        }
    )

    cat_labels = [CATEGORY_LABEL[c] for c in cat_rank.index]
    models = cat_rank.columns.tolist()
    n_categories = len(cat_labels)
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]

    max_rank = float(np.nanmax(cat_rank.to_numpy()))
    min_rank = float(np.nanmin(cat_rank.to_numpy()))
    outer = math.ceil(max_rank + 0.5)
    inner = max(1.0, math.floor(min_rank - 0.5))

    fig = plt.figure(figsize=(8.6, 8.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylim(outer, inner)
    grid_r = np.arange(inner, outer + 0.01, 1.0)
    ax.set_yticks(grid_r)
    ax.set_yticklabels([f"{r:.0f}" for r in grid_r], fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)

    palette = plt.get_cmap("tab10").colors
    model_color = {m: palette[i % len(palette)] for i, m in enumerate(MODEL_ORDER)}
    best_model_per_cat = cat_rank.idxmin(axis=1).to_dict()

    for model in models:
        vals = cat_rank[model].astype(float).tolist()
        vals_closed = vals + vals[:1]
        color = model_color.get(model, palette[0])
        ax.plot(
            angles,
            vals_closed,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=MODEL_LABEL.get(model, model),
            color=color,
            alpha=0.95,
        )
        ax.fill(angles, vals_closed, color=color, alpha=0.055)
        for j, cat in enumerate(cat_rank.index):
            if best_model_per_cat.get(cat) == model:
                ax.scatter(
                    angles[j],
                    vals[j],
                    s=86,
                    color=color,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=5,
                )

    split_label = "Cross-Subject" if split == "cross" else "Within-Subject"
    ax.set_title(f"Linear Full {split_label} Rank by Six EEG Categories", pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw six-category radar plot for 5.0 linear_full.")
    parser.add_argument(
        "--split",
        choices=["cross", "within"],
        default="cross",
        help="Use cross_subject or within_subject linear_full rank-by-type results.",
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    cat_rank = load_category_rank(args.input_dir, args.split)
    stem = f"linear_full_{args.split}_six_category"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cat_rank.rename(columns=MODEL_LABEL).to_csv(args.out_dir / f"{stem}_avg_rank.csv")
    plot_radar(cat_rank, args.out_dir, f"{stem}_radar", split=args.split)

    print(f"Split: {args.split}")
    print(f"Categories: {cat_rank.index.tolist()}")
    print(f"Models: {[MODEL_LABEL.get(m, m) for m in cat_rank.columns]}")
    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
