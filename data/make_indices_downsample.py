import os
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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

def generate_indices_downsample(
    data_dir,
    base_save_dir,
    split_ratio=(0.4, 0.3, 0.3),  # Train, Val, Test
    target_num=30,                # 固定每个Subject每个Label保留的样本数
    ratio_list=(0.01, 0.02, 0.05, 0.1, 0.2), # Few-shot ratios
    seed=42,
    custom_label_map=None
):
    rng = np.random.RandomState(seed)
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    
    cross_save_dir = os.path.join(base_save_dir, f"{dataset_name}_indices_cross_subject_downsample")
    within_save_dir = os.path.join(base_save_dir, f"{dataset_name}_indices_within_subject_downsample")
    os.makedirs(cross_save_dir, exist_ok=True)
    os.makedirs(within_save_dir, exist_ok=True)

    print(f"[Init] Scanning H5 files in {data_dir}...")
    print(f"[Strategy] Downsample to {target_num} samples per (Subject, Label).")


    temp_pool = {}
    h5_files = [os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs if f.endswith(".h5")]
    all_raw_labels = set()

    for file_path in tqdm(h5_files, ascii=True, desc="Scanning"):
        try:
            with h5py.File(file_path, "r") as f:
                raw_sid = f.attrs["subject_id"]
                sub_id = str(raw_sid.decode() if isinstance(raw_sid, bytes) else raw_sid)
                
                if sub_id not in temp_pool: temp_pool[sub_id] = {}

                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and name.endswith("eeg"):
                        l_attr = obj.attrs["label"]
                        lbl = int(l_attr[0] if hasattr(l_attr, '__len__') and not isinstance(l_attr, str) else l_attr)
                        if lbl not in temp_pool[sub_id]: temp_pool[sub_id][lbl] = []
                        temp_pool[sub_id][lbl].append((file_path, name, lbl))
                        all_raw_labels.add(lbl)
                f.visititems(collect)
        except Exception as e:
            print(f"Skipping broken file {file_path}: {e}")

    sorted_labels = sorted(list(all_raw_labels))
    final_map = custom_label_map if custom_label_map else {x: i for i, x in enumerate(sorted_labels)}
    print(f"[Labels] Found {len(sorted_labels)} unique labels. Map: {final_map}")


    clean_data = {} 
    
    for sub, label_dict in temp_pool.items():
        clean_data[sub] = {}
        for raw_lbl, items in label_dict.items():
            if raw_lbl not in final_map: continue
            mapped_lbl = final_map[raw_lbl]

            total_avail = len(items)
            keep_k = min(target_num, total_avail)

            indices = rng.choice(total_avail, size=keep_k, replace=False)
            selected_items = [items[i] for i in indices]

            lines = [f"{fp},{fn},{mapped_lbl}" for (fp, fn, _) in selected_items]
            clean_data[sub][mapped_lbl] = lines

    sub_ids = sorted(list(clean_data.keys()))
    if not sub_ids: raise RuntimeError("No valid subjects found.")

    # ==========================================
    # 3. Cross-Subject Split 
    # ==========================================
    print("\nGenerating Cross-subject indices (Downsampled)...")
    
    train_subs, temp_subs = train_test_split(sub_ids, test_size=(split_ratio[1] + split_ratio[2]), random_state=seed)
    val_test_size = split_ratio[1] + split_ratio[2]
    if val_test_size > 0:
        val_subs, test_subs = train_test_split(temp_subs, test_size=split_ratio[2]/val_test_size, random_state=seed)
    else:
        val_subs, test_subs = [], []

    cross_train_lines = []
    
    for name, subs in [("train", train_subs), ("val", val_subs), ("test", test_subs)]:
        lines = []
        for sid in subs:
            for lbl in clean_data[sid]:
                lines.extend(clean_data[sid][lbl])
        
        save_list_to_txt(lines, os.path.join(cross_save_dir, f"{name}_idx.txt"))
        if name == "train": cross_train_lines = lines

    if cross_train_lines and ratio_list:
        save_ratio_files(cross_train_lines, cross_save_dir, ratio_list, seed, "train_idx_ratio_")

    # ==========================================
    # 4. Within-Subject Split 
    # ==========================================
    print("\nGenerating Within-subject indices (Downsampled + Stratified)...")
    
    w_train_all, w_val_all, w_test_all = [], [], []

    for sid in sub_ids:
        for lbl, lines in clean_data[sid].items():
            if not lines: continue
            
            curr_lines = np.array(lines)
            rng.shuffle(curr_lines) 
            n = len(curr_lines)

            n_tr = int(n * split_ratio[0])
            n_va = int(n * split_ratio[1])
            
            w_train_all.extend(curr_lines[:n_tr])
            w_val_all.extend(curr_lines[n_tr : n_tr + n_va])
            w_test_all.extend(curr_lines[n_tr + n_va :])

    save_list_to_txt(w_train_all, os.path.join(within_save_dir, "train_idx.txt"))
    save_list_to_txt(w_val_all, os.path.join(within_save_dir, "val_idx.txt"))
    save_list_to_txt(w_test_all, os.path.join(within_save_dir, "test_idx.txt"))

    # Within Ratio-Shot
    if w_train_all and ratio_list:
        save_ratio_files(w_train_all, within_save_dir, ratio_list, seed, "train_idx_ratio_")

    print(f"\n[Done] Saved to:\n  Cross:  {cross_save_dir}\n  Within: {within_save_dir}")
    
if __name__ == "__main__":
    data_dir = "/zongsheng-group/ISRUC-Sleep_1/"
    base_save_dir = "indices"

    generate_indices_downsample(
        data_dir,
        base_save_dir,
        split_ratio=(0.4, 0.3, 0.3),  # Train, Val, Test
        target_num=40,                # 固定每个Subject每个Label保留的样本数
        ratio_list=(0.05, 0.1, 0.2,0.3), # Few-shot ratios
        seed=42,
        custom_label_map=None
    )