from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path("/benchmark-eeg/5.0_version")
TABLE_DIR = REPO_ROOT / "tables" / "results"
OUT_DIR = REPO_ROOT / "analysis" / "six_category_radar"

CATEGORY_ORDER = ["Type-I", "Type-II", "Type-III", "Type-IV", "Type-V", "Type-VI"]
CATEGORY_LABEL = {
    "Type-I": "Signal\nReliability",
    "Type-II": "Biometrics\n& Disease",
    "Type-III": "Consciousness\n& State",
    "Type-IV": "Cognition\n& Emotion",
    "Type-V": "Naturalistic\nStimulus Decoding",
    "Type-VI": "Motor\n& Interaction",
}

MODEL_ORDER = [
    "BrainOmni",
    "LaBraM",
    "BIOT",
    "FEMBA",
    "NeuroLM",
    "CBraMod",
    "NeuroGPT",
    "REVE",
    "EEGMamba",
    "BENDR",
]

MODEL_ALIASES = {
    "brainomni": "BrainOmni",
    "labram": "LaBraM",
    "labr am": "LaBraM",
    "biot": "BIOT",
    "femba": "FEMBA",
    "neurolm": "NeuroLM",
    "cbramod": "CBraMod",
    "neurogpt": "NeuroGPT",
    "reve": "REVE",
    "revenew": "REVE",
    "reve_new": "REVE",
    "eegmamba": "EEGMamba",
    "bendr": "BENDR",
}


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("extraversial", "extraversion")
    text = re.sub(r"_wsn$", "", text)
    text = re.sub(r"_balanced$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def canonical_model(col: str) -> str | None:
    key = str(col).strip()
    if key in MODEL_ORDER:
        return key
    return MODEL_ALIASES.get(key.lower())


def model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if canonical_model(c) is not None]


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """After alias normalization, merge duplicate model columns left-to-right."""
    if not df.columns.has_duplicates:
        return df
    out = pd.DataFrame(index=df.index)
    for col in dict.fromkeys(df.columns):
        values = df.loc[:, df.columns == col]
        if values.shape[1] == 1:
            out[col] = values.iloc[:, 0]
        else:
            out[col] = values.bfill(axis=1).iloc[:, 0]
    return out


def category_map(acc_df: pd.DataFrame) -> pd.DataFrame:
    required = {"Category", "dataset_display"}
    missing = required - set(acc_df.columns)
    if missing:
        raise ValueError(f"Category table missing columns: {sorted(missing)}")
    out = acc_df[["Category", "dataset_display"]].copy()
    out["dataset_key"] = out["dataset_display"].map(normalize_text)
    out = out[out["Category"].isin(CATEGORY_ORDER)].drop_duplicates("dataset_key")
    return out


def ranks_from_rank_table(rank_path: Path, category_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_df = pd.read_csv(rank_path)
    cat_df = category_map(pd.read_csv(category_path))
    rank_df["dataset_key"] = rank_df["dataset_display"].map(normalize_text)
    merged = rank_df.merge(cat_df[["dataset_key", "Category"]], on="dataset_key", how="left")
    merged = merged[merged["Category"].isin(CATEGORY_ORDER)].copy()

    rename = {c: canonical_model(c) for c in model_columns(merged)}
    merged = merged.rename(columns=rename)
    merged = coalesce_duplicate_columns(merged)
    rank_cols = [m for m in MODEL_ORDER if m in merged.columns]
    long_df = merged.melt(
        id_vars=["Category", "dataset_display"],
        value_vars=rank_cols,
        var_name="model",
        value_name="rank",
    ).dropna(subset=["rank"])
    cat_rank = merged.groupby("Category")[rank_cols].mean(numeric_only=True).reindex(CATEGORY_ORDER)
    cat_rank = cat_rank.dropna(how="all")
    cat_rank = cat_rank[[c for c in MODEL_ORDER if c in cat_rank.columns]]
    return long_df, cat_rank


def ranks_from_acc_table(acc_path: Path, exclude_below_chance: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc_df = pd.read_csv(acc_path)
    acc_df = acc_df[acc_df["Category"].isin(CATEGORY_ORDER)].copy()

    rename = {c: canonical_model(c) for c in model_columns(acc_df)}
    acc_df = acc_df.rename(columns=rename)
    acc_df = coalesce_duplicate_columns(acc_df)
    acc_cols = [m for m in MODEL_ORDER if m in acc_df.columns]

    mean_df = acc_df[["Category", "dataset_display", "chance_level", *acc_cols]].copy()
    if exclude_below_chance and "chance_level" in mean_df.columns:
        for col in acc_cols:
            below = mean_df["chance_level"].notna() & mean_df[col].notna() & (mean_df[col] < mean_df["chance_level"])
            mean_df.loc[below, col] = np.nan

    rank_only = mean_df[acc_cols].rank(axis=1, ascending=False, method="average", na_option="bottom")
    rank_df = pd.concat([mean_df[["Category", "dataset_display"]], rank_only], axis=1)
    long_df = rank_df.melt(
        id_vars=["Category", "dataset_display"],
        value_vars=acc_cols,
        var_name="model",
        value_name="rank",
    ).dropna(subset=["rank"])
    cat_rank = rank_df.groupby("Category")[acc_cols].mean(numeric_only=True).reindex(CATEGORY_ORDER)
    cat_rank = cat_rank.dropna(how="all")
    return mean_df, long_df, cat_rank


def plot_radar(cat_rank: pd.DataFrame, out_dir: Path, stem: str, title: str) -> None:
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

    labels = [CATEGORY_LABEL[c] for c in cat_rank.index]
    models = cat_rank.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    values = cat_rank.to_numpy(dtype=float)
    outer = math.ceil(float(np.nanmax(values)) + 0.5)
    inner = max(1.0, math.floor(float(np.nanmin(values)) - 0.5))

    fig = plt.figure(figsize=(8.6, 8.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(outer, inner)
    ticks = np.arange(inner, outer + 0.01, 1.0)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{x:.0f}" for x in ticks], fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)

    palette = plt.get_cmap("tab10").colors
    colors = {m: palette[i % len(palette)] for i, m in enumerate(MODEL_ORDER)}
    best_model = cat_rank.idxmin(axis=1).to_dict()
    for model in models:
        vals = cat_rank[model].astype(float).tolist()
        closed = vals + vals[:1]
        color = colors.get(model, palette[0])
        ax.plot(angles, closed, linewidth=2.0, marker="o", markersize=4.5, label=model, color=color)
        ax.fill(angles, closed, color=color, alpha=0.055)
        for idx, cat in enumerate(cat_rank.index):
            if best_model.get(cat) == model:
                ax.scatter(angles[idx], vals[idx], s=86, color=color, edgecolors="black", linewidths=0.8, zorder=5)

    ax.set_title(title, pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw six-category radar plots from tables/results CSV files.")
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-exclude-below-chance", action="store_true")
    args = parser.parse_args()

    cross_long, cross_cat_rank = ranks_from_rank_table(
        args.table_dir / "rank_df_for_cross_subject.csv",
        args.table_dir / "acc_for_cross_subject.csv",
    )
    cross_long.to_csv(args.out_dir / "tables_results_cross_rank_long.csv", index=False)
    cross_cat_rank.to_csv(args.out_dir / "tables_results_cross_six_category_avg_rank.csv")
    plot_radar(
        cross_cat_rank,
        args.out_dir,
        "tables_results_cross_six_category_radar",
        "Cross-Subject Rank by Six EEG Categories",
    )

    within_mean, within_long, within_cat_rank = ranks_from_acc_table(
        args.table_dir / "acc_for_within.csv",
        exclude_below_chance=not args.no_exclude_below_chance,
    )
    within_mean.to_csv(args.out_dir / "tables_results_within_dataset_model_accuracy.csv", index=False)
    within_long.to_csv(args.out_dir / "tables_results_within_rank_long.csv", index=False)
    within_cat_rank.to_csv(args.out_dir / "tables_results_within_six_category_avg_rank.csv")
    plot_radar(
        within_cat_rank,
        args.out_dir,
        "tables_results_within_six_category_radar",
        "Within-Subject Rank by Six EEG Categories",
    )

    print(f"Cross categories: {cross_cat_rank.shape[0]}, models: {cross_cat_rank.shape[1]}")
    print(f"Within categories: {within_cat_rank.shape[0]}, models: {within_cat_rank.shape[1]}")
    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
