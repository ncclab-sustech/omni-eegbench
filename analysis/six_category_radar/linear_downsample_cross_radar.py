from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path("/benchmark-eeg/5.0_version")
RESULT_ROOT = REPO_ROOT / "result_5" / "linear_downsample"
REFERENCE_CSV = REPO_ROOT / "experiment_tracking" / "experiment_reference_updated.csv"
LABEL_CSV = REPO_ROOT / "experiment_tracking" / "experiment_reference_with_label.csv"
OUT_DIR = REPO_ROOT / "analysis" / "six_category_radar"

SEEDS = [42, 10, 5]
METRIC_KEY = "accuracy"

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

# Current tracking uses Type-I ... Type-VII. Type-VII is treated as the
# unknown/mixed bucket for this figure and is excluded.
CATEGORY_ORDER = ["Type-I", "Type-II", "Type-III", "Type-IV", "Type-V", "Type-VI"]
CATEGORY_LABEL = {
    "Type-I": "Signal\nReliability",
    "Type-II": "Biometrics\n& Disease",
    "Type-III": "Consciousness\n& State",
    "Type-IV": "Cognition\n& Emotion",
    "Type-V": "Naturalistic\nStimulus Decoding",
    "Type-VI": "Motor\n& Interaction",
}


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def normalize_name(value: str) -> str:
    key = clean(value).lower()
    replacements = {
        "–": "_",
        "—": "_",
        "-": "_",
        " ": "_",
        "（": "(",
        "）": ")",
        "monitering": "monitoring",
        "extraversial": "extraversion",
        "fintuen": "finetune",
    }
    for src, dst in replacements.items():
        key = key.replace(src, dst)
    key = re.sub(r"\([^)]*\)", "", key)
    key = re.sub(r"_balanced$", "", key)
    key = re.sub(r"_old_badscale$", "", key)
    key = re.sub(r"_wsn$", "", key)
    return re.sub(r"[^a-z0-9]+", "", key)


def names_match(query: str, candidate: str) -> bool:
    q = normalize_name(query)
    c = normalize_name(candidate)
    if not q or not c:
        return False
    return q == c or (len(q) >= 6 and q in c) or (len(c) >= 6 and c in q)


def result_dataset_from_dir(name: str, mode_name: str) -> str:
    marker = f"_{mode_name}_"
    return name.split(marker, 1)[0] if marker in name else name


def load_chance_levels(path: Path) -> dict[str, float]:
    """Load num_labels from experiment_reference_with_label.csv and return dataset_key -> 1/num_labels."""
    label_df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    mapping = {}
    for _, row in label_df.iterrows():
        try:
            n = int(row["num_labels"])
            if n > 0:
                mapping[normalize_name(clean(row["dataset"]))] = 1.0 / n
        except (ValueError, KeyError):
            pass
    return mapping


def load_reference(path: Path, task_col: str, label_path: Path = LABEL_CSV) -> pd.DataFrame:
    ref = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    ref["dataset_display"] = ref["dataset"].map(clean)
    ref["dataset_exp_name"] = ref.get("dataset_exp_name", "").map(clean)
    ref["category"] = ref["category"].map(clean)
    if task_col not in ref.columns:
        raise KeyError(f"Column not found in reference CSV: {task_col}")
    status = ref[task_col].map(clean).str.upper()
    applicable = ref[(status != "N/A") & (status != "")].copy()
    applicable = applicable[applicable["category"].isin(CATEGORY_ORDER)].copy()
    applicable["dataset_key"] = applicable["dataset_display"].map(normalize_name)
    chance_map = load_chance_levels(label_path)
    applicable["chance_level"] = applicable["dataset_key"].map(chance_map)
    return applicable


def discover_result_dirs(result_root: Path, mode_name: str) -> pd.DataFrame:
    rows = []
    if not result_root.exists():
        return pd.DataFrame(columns=["result_dir", "result_dir_name", "result_dataset", "result_key"])
    for path in result_root.iterdir():
        if not path.is_dir() or f"_{mode_name}_" not in path.name:
            continue
        result_dataset = result_dataset_from_dir(path.name, mode_name)
        rows.append(
            {
                "result_dir": path,
                "result_dir_name": path.name,
                "result_dataset": result_dataset,
                "result_key": normalize_name(result_dataset),
            }
        )
    return pd.DataFrame(rows)


def choose_result_dir(row: pd.Series, result_dirs: pd.DataFrame, mode_name: str) -> Path | None:
    if result_dirs.empty:
        return None

    queries = [row["dataset_exp_name"], row["dataset_display"]]
    queries = [q for q in queries if clean(q)]

    for query in queries:
        exact = result_dirs[result_dirs["result_key"] == normalize_name(query)]
        if len(exact):
            return exact.sort_values("result_dir_name").iloc[0]["result_dir"]

    fuzzy_parts = []
    for query in queries:
        fuzzy = result_dirs[result_dirs["result_dataset"].map(lambda x: names_match(query, x))]
        if len(fuzzy):
            fuzzy_parts.append(fuzzy)

    if not fuzzy_parts:
        return None

    fuzzy = pd.concat(fuzzy_parts).drop_duplicates("result_dir_name").copy()
    preferred_exp = normalize_name(row["dataset_exp_name"])

    def score(name: str) -> tuple[int, int, int, str]:
        dataset = result_dataset_from_dir(name, mode_name)
        exact_exp = 0 if preferred_exp and normalize_name(dataset) == preferred_exp else 1
        suffix = 0 if name.endswith("_linear_downsample") else 1
        return exact_exp, suffix, len(name), name

    fuzzy["score"] = fuzzy["result_dir_name"].map(score)
    return fuzzy.sort_values("score").iloc[0]["result_dir"]


def read_summary(summary_path: Path, allow_model: str | None = None) -> tuple[str, float] | None:
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return None
    model = clean(summary.get("model") or summary_path.parent.name).lower()
    metrics = summary.get("test_metrics", {})
    value = metrics.get(METRIC_KEY)
    if value is None:
        value = metrics.get("balanced_accuracy")
    if value is None:
        return None
    # allow_model lets callers override the MODEL_ORDER check (e.g. reve_new -> reve)
    if model not in MODEL_ORDER and model != allow_model:
        return None
    return model, float(value)


def collect_results(matched: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in matched.iterrows():
        result_dir = row["result_dir"]
        if not isinstance(result_dir, Path):
            continue
        for seed in SEEDS:
            ratio_dir = result_dir / f"seed_{seed}_downsample_t40" / "ratio_full"
            if not ratio_dir.exists():
                continue

            # All models except reve/reve_new
            for summary_path in ratio_dir.glob("*/summary.json"):
                if summary_path.parent.name in ("reve", "reve_new"):
                    continue
                item = read_summary(summary_path)
                if item is None:
                    continue
                model, accuracy = item
                records.append(
                    {
                        "category": row["category"],
                        "category_label": CATEGORY_LABEL[row["category"]].replace("\n", " "),
                        "dataset": row["dataset_display"],
                        "dataset_exp_name": row["dataset_exp_name"],
                        "result_dir": result_dir.name,
                        "seed": seed,
                        "model": model,
                        "model_label": MODEL_LABEL[model],
                        "accuracy": accuracy,
                        "chance_level": row.get("chance_level"),
                        "summary_path": str(summary_path),
                    }
                )

            # reve: prefer reve_new if available, else fall back to reve
            reve_new_path = ratio_dir / "reve_new" / "summary.json"
            reve_path = ratio_dir / "reve" / "summary.json"
            reve_source = reve_new_path if reve_new_path.exists() else reve_path
            if reve_source.exists():
                item = read_summary(reve_source, allow_model="reve_new")
                if item is not None:
                    _, accuracy = item
                    records.append(
                        {
                            "category": row["category"],
                            "category_label": CATEGORY_LABEL[row["category"]].replace("\n", " "),
                            "dataset": row["dataset_display"],
                            "dataset_exp_name": row["dataset_exp_name"],
                            "result_dir": result_dir.name,
                            "seed": seed,
                            "model": "reve",
                            "model_label": MODEL_LABEL["reve"],
                            "accuracy": accuracy,
                            "chance_level": row.get("chance_level"),
                            "summary_path": str(reve_source),
                        }
                    )

    return pd.DataFrame(records)


def build_category_rank(
    long_df: pd.DataFrame,
    exclude_below_chance: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_df = (
        long_df.groupby(["category", "dataset", "model"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), n_seed=("seed", "nunique"),
             chance_level=("chance_level", "first"))
    )

    if exclude_below_chance:
        # Mask results below chance level so they are excluded from ranking
        below_chance = (
            mean_df["chance_level"].notna()
            & (mean_df["mean_accuracy"] < mean_df["chance_level"])
        )
        n_below = below_chance.sum()
        if n_below:
            print(f"[below-chance] excluding {n_below} (dataset, model) pairs from ranking "
                  f"(mean_accuracy < chance_level)")
            mean_df.loc[below_chance, "mean_accuracy"] = float("nan")
    else:
        print("[below-chance] exclusion disabled; all results included in ranking")

    acc_pivot = mean_df.pivot(index="dataset", columns="model", values="mean_accuracy")
    category_by_dataset = mean_df.drop_duplicates("dataset").set_index("dataset")["category"]

    # na_option="bottom" pushes below-chance models to worst rank within that dataset
    rank_df = acc_pivot.rank(axis=1, ascending=False, method="average", na_option="bottom")
    rank_df["category"] = category_by_dataset
    cat_rank = rank_df.groupby("category").mean(numeric_only=True).reindex(CATEGORY_ORDER)
    cat_rank = cat_rank.dropna(how="all")

    model_order = cat_rank.mean(axis=0, skipna=True).sort_values().index.tolist()
    cat_rank = cat_rank[model_order]
    return mean_df, rank_df, cat_rank


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
    # Fix colors to MODEL_ORDER positions so cross/within plots are consistent
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
    ax.set_title(f"Linear Downsample {split_label} Rank by Six EEG Categories", pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw six-category radar plot for 5.0 linear_downsample.")
    parser.add_argument(
        "--split",
        choices=["cross", "within"],
        default="cross",
        help="Use cross_subject or within_subject linear_downsample results.",
    )
    parser.add_argument("--reference", type=Path, default=REFERENCE_CSV)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--exclude-below-chance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude (dataset, model) pairs below chance level from ranking (default: enabled).",
    )
    args = parser.parse_args()

    mode_name = "cross_subject" if args.split == "cross" else "within_subject"
    task_col = f"linear_downsample_{args.split}"
    stem = f"linear_downsample_{args.split}_six_category"

    ref = load_reference(args.reference, task_col)
    # Disease datasets (Type-II) are excluded for within-subject evaluation
    if args.split == "within":
        ref = ref[ref["category"] != "Type-II"].copy()
        print("[within] Type-II (Biometrics & Disease) excluded from ranking")
    result_dirs = discover_result_dirs(args.result_root, mode_name)
    matched = ref.copy()
    matched["result_dir"] = matched.apply(lambda row: choose_result_dir(row, result_dirs, mode_name), axis=1)
    matched["result_dir_name"] = matched["result_dir"].map(lambda p: p.name if isinstance(p, Path) else "")

    long_df = collect_results(matched)
    if long_df.empty:
        raise RuntimeError(f"No {METRIC_KEY} results found under {args.result_root}")

    mean_df, rank_df, cat_rank = build_category_rank(long_df, exclude_below_chance=args.exclude_below_chance)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.out_dir / f"{stem}_long.csv", index=False)
    mean_df.to_csv(args.out_dir / f"{stem}_dataset_model_mean.csv", index=False)
    cat_rank.rename(columns=MODEL_LABEL).to_csv(args.out_dir / f"{stem}_avg_rank.csv")
    matched[["category", "dataset_display", "dataset_exp_name", task_col, "result_dir_name"]].to_csv(
        args.out_dir / f"{stem}_matched_datasets.csv",
        index=False,
    )

    plot_radar(cat_rank, args.out_dir, f"{stem}_radar", split=args.split)

    print(f"Split: {args.split} ({mode_name})")
    print(f"Reference rows used: {len(ref)}")
    print(f"Matched datasets: {matched['result_dir'].notna().sum()} / {len(matched)}")
    print(f"Datasets with plotted results: {long_df['dataset'].nunique()}")
    print(f"Models: {[MODEL_LABEL.get(m, m) for m in cat_rank.columns]}")
    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
