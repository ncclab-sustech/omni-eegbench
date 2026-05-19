from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from six_category_radar_no_caption import (
    CATEGORY_ORDER,
    LABEL_CSV,
    MODEL_LABEL,
    MODEL_ORDER,
    OUT_DIR,
    REFERENCE_CSV,
    build_category_rank,
    clean,
    load_chance_levels,
    names_match,
    normalize_name,
    plot_radar,
)


REPO_ROOT = Path("/benchmark-eeg/5.0_version")
TABLE_DIR = REPO_ROOT / "tables" / "withstd"
DEFAULT_CSV = {
    "cross": TABLE_DIR / "benchmark_results_linear_downsample_cross_subject_mean_var.csv",
    "within": TABLE_DIR / "benchmark_results_linear_downsample_within_subject_mean_var.csv",
}


def parse_mean(value) -> float:
    text = clean(value)
    if not text or text == "--":
        return float("nan")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        return float("nan")
    return float(match.group(0))


def load_category_reference(reference_path: Path, label_path: Path = LABEL_CSV) -> pd.DataFrame:
    chance_map = load_chance_levels(label_path)

    def load_rows(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        frame["dataset_display"] = frame["dataset"].map(clean)
        frame["dataset_exp_name"] = frame.get("dataset_exp_name", "").map(clean)
        frame["category"] = frame["category"].map(clean)
        frame = frame[frame["category"].isin(CATEGORY_ORDER)].copy()
        frame["dataset_key"] = frame["dataset_display"].map(normalize_name)
        frame["chance_level"] = frame["dataset_key"].map(chance_map)
        return frame

    ref = load_rows(reference_path)
    label_ref = load_rows(label_path)
    known_keys = set(ref["dataset_key"])
    fallback_rows = label_ref[~label_ref["dataset_key"].isin(known_keys)].copy()
    return pd.concat([ref, fallback_rows], ignore_index=True)


def build_reference_lookup(ref: pd.DataFrame) -> tuple[dict[str, pd.Series], list[pd.Series]]:
    lookup = {}
    rows = []
    for _, row in ref.iterrows():
        rows.append(row)
        for col in ("dataset_display", "dataset_exp_name"):
            key = normalize_name(row.get(col, ""))
            if key and key not in lookup:
                lookup[key] = row
    return lookup, rows


def find_reference_row(dataset_name: str, lookup: dict[str, pd.Series], rows: list[pd.Series]) -> pd.Series | None:
    key = normalize_name(dataset_name)
    if key in lookup:
        return lookup[key]
    matches = [
        row
        for row in rows
        if names_match(dataset_name, row.get("dataset_display", ""))
        or names_match(dataset_name, row.get("dataset_exp_name", ""))
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda row: len(clean(row.get("dataset_exp_name", ""))))[-1]


def load_csv_results(csv_path: Path, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    wide = pd.read_csv(csv_path, index_col=0, encoding="utf-8-sig", keep_default_na=False)
    wide.index = wide.index.map(clean)

    if "reve_new" in wide.columns:
        if "reve" not in wide.columns:
            wide["reve"] = wide["reve_new"]
        else:
            reve_new_mean = wide["reve_new"].map(parse_mean)
            has_reve_new = reve_new_mean.notna()
            wide.loc[has_reve_new, "reve"] = wide.loc[has_reve_new, "reve_new"]

    model_cols = [model for model in MODEL_ORDER if model in wide.columns]
    if not model_cols:
        raise RuntimeError(f"No known model columns found in {csv_path}")

    ref_lookup, ref_rows = build_reference_lookup(ref)
    records = []
    matched_rows = []
    unmatched = []
    n_data_rows = 0

    for dataset_name, row in wide.iterrows():
        if normalize_name(dataset_name) in {"mean", "std", "n"}:
            continue
        n_data_rows += 1
        ref_row = find_reference_row(dataset_name, ref_lookup, ref_rows)
        if ref_row is None:
            unmatched.append(dataset_name)
            continue

        matched_rows.append(
            {
                "category": ref_row["category"],
                "dataset_display": ref_row["dataset_display"],
                "dataset_exp_name": ref_row["dataset_exp_name"],
                "csv_dataset": dataset_name,
            }
        )
        for model in model_cols:
            accuracy = parse_mean(row[model])
            if np.isnan(accuracy):
                continue
            records.append(
                {
                    "category": ref_row["category"],
                    "dataset": ref_row["dataset_display"],
                    "dataset_exp_name": ref_row["dataset_exp_name"],
                    "csv_dataset": dataset_name,
                    "seed": 0,
                    "model": model,
                    "model_label": MODEL_LABEL[model],
                    "accuracy": accuracy,
                    "chance_level": ref_row.get("chance_level"),
                    "source_csv": str(csv_path),
                }
            )

    if unmatched:
        print(f"[match] {len(unmatched)} CSV datasets were not found in the reference and were skipped:")
        print(", ".join(unmatched))

    long_df = pd.DataFrame(records)
    matched_df = pd.DataFrame(matched_rows).drop_duplicates("csv_dataset")
    return long_df, matched_df, n_data_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw six-category radar plot from withstd mean/variance CSV tables."
    )
    parser.add_argument(
        "--split",
        choices=["cross", "within", "both"],
        default="both",
        help="Which CSV table to plot.",
    )
    parser.add_argument("--cross-csv", type=Path, default=DEFAULT_CSV["cross"])
    parser.add_argument("--within-csv", type=Path, default=DEFAULT_CSV["within"])
    parser.add_argument("--reference", type=Path, default=REFERENCE_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--exclude-below-chance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude (dataset, model) pairs below chance level from ranking (default: enabled).",
    )
    args = parser.parse_args()

    splits = ["cross", "within"] if args.split == "both" else [args.split]
    csv_by_split = {"cross": args.cross_csv, "within": args.within_csv}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        stem = f"linear_downsample_{split}_six_category_from_csv"
        ref = load_category_reference(args.reference, LABEL_CSV)
        if split == "within":
            ref = ref[ref["category"] != "Type-II"].copy()
            print("[within] Type-II (Biometrics & Disease) excluded from ranking")

        long_df, matched_df, n_data_rows = load_csv_results(csv_by_split[split], ref)
        if long_df.empty:
            raise RuntimeError(f"No plotted results found in {csv_by_split[split]}")

        mean_df, rank_df, cat_rank = build_category_rank(
            long_df,
            exclude_below_chance=args.exclude_below_chance,
        )
        cat_rank = cat_rank.reindex(CATEGORY_ORDER).dropna(how="all")

        long_df.to_csv(args.out_dir / f"{stem}_long.csv", index=False)
        mean_df.to_csv(args.out_dir / f"{stem}_dataset_model_mean.csv", index=False)
        rank_df.to_csv(args.out_dir / f"{stem}_dataset_model_rank.csv")
        cat_rank.rename(columns=MODEL_LABEL).to_csv(args.out_dir / f"{stem}_avg_rank.csv")
        matched_df.to_csv(args.out_dir / f"{stem}_matched_datasets.csv", index=False)

        plot_radar(cat_rank, args.out_dir, f"{stem}_radar", split=split)

        print(f"Split: {split}")
        print(f"CSV: {csv_by_split[split]}")
        print(f"Reference category rows used: {len(ref)}")
        print(f"Matched CSV datasets: {matched_df['csv_dataset'].nunique()} / {n_data_rows}")
        print(f"Datasets with plotted results: {long_df['dataset'].nunique()}")
        print(f"Models: {[MODEL_LABEL.get(m, m) for m in cat_rank.columns]}")
        print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
