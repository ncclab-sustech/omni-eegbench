import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def save_list_to_txt(data_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for line in data_list:
            f.write(line + "\n")


def _parse_label_from_line(line: str) -> int:
    """
    line format: file_path,internal_path,label
    label is the last field after the last comma
    """
    return int(line.rsplit(",", 1)[-1])


def _ratio_to_tag(r: float) -> str:
    """
    0.02 -> "0p02"
    0.1  -> "0p1"
    """
    s = str(float(r))
    return s.replace(".", "p")


def make_ratio_train_lines(
    train_lines,
    ratio_list=(0.02, 0.05, 0.1, 0.3, 0.5),
    seed=42,
    min_per_class=1,
):
    """
    From full train_lines, sample ceil(ratio * n_class) examples per class for each ratio.

    Returns:
      dict: {ratio(float): sampled_lines_list}
    """
    rng = np.random.RandomState(seed)

    # group by class
    by_class = {}
    for line in train_lines:
        y = _parse_label_from_line(line)
        by_class.setdefault(y, []).append(line)

    classes = sorted(by_class.keys())
    out = {}

    for r in ratio_list:
        r = float(r)
        if r <= 0:
            raise ValueError(f"[ratio-shot] ratio must be > 0, got {r}")

        chosen = []
        for y in classes:
            lines_y = by_class[y]
            n_y = len(lines_y)
            if n_y == 0:
                continue

            # sample size per class
            k = int(np.ceil(r * n_y))
            k = max(int(min_per_class), k)
            k = min(k, n_y)

            idx = rng.choice(n_y, size=k, replace=False)
            chosen.extend([lines_y[i] for i in idx])

        rng.shuffle(chosen)
        out[r] = chosen

    return out


def save_ratio_files(
    train_lines,
    save_dir,
    ratio_list=(0.02, 0.05, 0.1, 0.3, 0.5),
    seed=42,
    prefix="train_idx_ratio_",
):
    """
    Generate and save ratio-shot train idx files under save_dir.
    Files: train_idx_ratio_0p02.txt, train_idx_ratio_0p05.txt, ...
    """
    ratio_dict = make_ratio_train_lines(train_lines, ratio_list=ratio_list, seed=seed)
    for r, lines in ratio_dict.items():
        tag = _ratio_to_tag(r)
        fp = os.path.join(save_dir, f"{prefix}{tag}.txt")
        save_list_to_txt(lines, fp)


def generate_indices(
    data_dir,
    base_save_dir,
    split_ratio=(0.8, 0.1, 0.1),
    # ratio_list=(0.02, 0.05, 0.1, 0.3, 0.5),
    ratio_list=(0.01,0.02,0.05,0.1,0.2),
    ratio_seed=42,
    custom_label_map=None
):
    """
    Args:
        data_dir:  Directory of H5 files.
        base_save_dir:  Base directory to save indices.  
        split_ratio: (train, val, test) / Ratios for splitting.
        ratio_list: tuple/list of ratios for ratio-shot train generation
        ratio_seed: random seed for ratio-shot sampling
    """
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    cross_save_dir = os.path.join(base_save_dir, f"{dataset_name}_indices_cross_subject")
    within_save_dir = os.path.join(base_save_dir, f"{dataset_name}_indices_within_subject")

    os.makedirs(cross_save_dir, exist_ok=True)
    os.makedirs(within_save_dir, exist_ok=True)

    # Storage: {subject_id: [line1, line2, ...]}
    subjects_data_raw = {}
    all_unique_labels = set()
    h5_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(root, file))

    print(f"Start scanning {len(h5_files)} H5 files...")

    for file_path in tqdm(h5_files, desc="Scanning H5 Files"):
        try:
            with h5py.File(file_path, "r") as f:
                sub_id_raw = f.attrs["subject_id"]
                sub_id = str(sub_id_raw) if not isinstance(sub_id_raw, (bytes, np.bytes_)) else sub_id_raw.decode()
                
                if sub_id not in subjects_data_raw:
                    subjects_data_raw[sub_id] = []

                def collect_segments(name, obj):
                    if isinstance(obj, h5py.Dataset) and name.endswith("eeg"):
                        raw_label = obj.attrs["label"]
                        if hasattr(raw_label, '__len__') and not isinstance(raw_label, str):
                            raw_val = raw_label[0]
                        else:
                            raw_val = raw_label
                        
                        raw_val_int = int(raw_val)

                        subjects_data_raw[sub_id].append((file_path, name, raw_val_int))
                        all_unique_labels.add(raw_val_int)

                f.visititems(collect_segments)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    sorted_labels = sorted(list(all_unique_labels))
    min_lbl = sorted_labels[0] if sorted_labels else 0
    max_lbl = sorted_labels[-1] if sorted_labels else 0
    num_classes = len(sorted_labels)
    
    print(f"\n[Label Analysis] Found {num_classes} unique labels: {sorted_labels}")

    final_map = {} 

    if custom_label_map is not None:
        print("-> Using User Provided Map.")
        final_map = custom_label_map
    
    else:
        final_map = {i: i for i in sorted_labels}

    # elif sorted_labels == list(range(num_classes)):
    #     print("-> Detected 0-indexed continuous labels. Using as is.")
    #     final_map = {i: i for i in sorted_labels}

    # elif sorted_labels == list(range(1, num_classes + 1)):
    #     print("-> Detected 1-indexed continuous labels. Applying (label - 1).")
    #     final_map = {i: i - 1 for i in sorted_labels}

    # else:
    #     error_msg = (
    #         f"\n\n!!! CRITICAL DATA ERROR !!!\n"
    #         f"Labels are discontinuous or non-standard.\n"
    #         f"Found Labels: {sorted_labels}\n"
    #         f"Cannot automatically determine mapping.\n"
    #         f"Please provide 'custom_label_map' argument in main code.\n"
    #         f"Example: custom_label_map = {{{sorted_labels[0]}: 0, {sorted_labels[-1]}: 1, ...}}\n"
    #     )
    #     raise ValueError(error_msg)

    subjects_data = {} 
    
    total_valid_samples = 0
    for sub_id, raw_items in subjects_data_raw.items():
        subjects_data[sub_id] = []
        for (f_path, f_name, r_val) in raw_items:
            if r_val in final_map:
                final_lbl = final_map[r_val]
                line = f"{f_path},{f_name},{final_lbl}"
                subjects_data[sub_id].append(line)
                total_valid_samples += 1
            else:
                pass

    # print(f"Total processed samples: {total_valid_samples}")
    # if total_valid_samples == 0:
    #     raise RuntimeError("No valid samples produced after mapping!")

    sub_ids = sorted(list(subjects_data.keys()))

    if len(sub_ids) == 0:
        raise RuntimeError("No subject_id found in scanned H5 files.")

    # --- Cross-subject Split ---
    print("Generating Cross-subject indices...")
    train_subs, temp_subs = train_test_split(
        sub_ids, test_size=(split_ratio[1] + split_ratio[2]), random_state=42
    )
    val_subs, test_subs = train_test_split(
        temp_subs,
        test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
        random_state=42,
    )

    cross_train_lines = None
    for name, subs in [("train", train_subs), ("val", val_subs), ("test", test_subs)]:
        combined_list = []
        for sid in subs:
            combined_list.extend(subjects_data[sid])
        save_list_to_txt(combined_list, os.path.join(cross_save_dir, f"{name}_idx.txt"))
        if name == "train":
            cross_train_lines = combined_list

    if cross_train_lines is not None and ratio_list:
        print(f"Generating Cross-subject ratio-shot train indices: {list(ratio_list)} ...")
        save_ratio_files(
            cross_train_lines,
            cross_save_dir,
            ratio_list=ratio_list,
            seed=ratio_seed,
            prefix="train_idx_ratio_",
        )

    # --- Within-subject Split ---
    print("Generating Within-subject indices...")
    w_train, w_val, w_test = [], [], []

    for sid in sub_ids:
        samples = subjects_data[sid]
        if len(samples) < 3:
            continue

        t_s, tmp_s = train_test_split(
            samples, test_size=(split_ratio[1] + split_ratio[2]), random_state=42
        )
        v_s, ts_s = train_test_split(
            tmp_s,
            test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
            random_state=42,
        )
        w_train.extend(t_s)
        w_val.extend(v_s)
        w_test.extend(ts_s)

    save_list_to_txt(w_train, os.path.join(within_save_dir, "train_idx.txt"))
    save_list_to_txt(w_val, os.path.join(within_save_dir, "val_idx.txt"))
    save_list_to_txt(w_test, os.path.join(within_save_dir, "test_idx.txt"))

    if w_train and ratio_list:
        print(f"Generating Within-subject ratio-shot train indices: {list(ratio_list)} ...")
        save_ratio_files(
            w_train,
            within_save_dir,
            ratio_list=ratio_list,
            seed=ratio_seed,
            prefix="train_idx_ratio_",
        )

    print("Success: All index files generated.")
    print(f"Cross-subject path: {cross_save_dir}")
    print(f"Within-subject path: {within_save_dir}")


if __name__ == "__main__":
    data_path = "/zongsheng-group/EEG-IO/"
    save_path = "indices"

    generate_indices(
        data_path,
        save_path,
        split_ratio=(0.4, 0.3, 0.3),
        ratio_list=(0.05, 0.1, 0.3),
        ratio_seed=42,
        custom_label_map=None
    )
