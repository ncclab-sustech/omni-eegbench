import os
import h5py
import numpy as np
import re
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
    Nested ratio-shot sampling:
    for each class, smaller-ratio subset is always contained in larger-ratio subset.

    Returns:
      dict: {ratio(float): sampled_lines_list}
    """
    rng = np.random.RandomState(seed)
    ratio_list = sorted([float(r) for r in ratio_list])

    # group by class
    by_class = {}
    for line in train_lines:
        y = _parse_label_from_line(line)
        by_class.setdefault(y, []).append(line)

    classes = sorted(by_class.keys())


    shuffled_by_class = {}
    for y in classes:
        lines_y = list(by_class[y])
        perm = rng.permutation(len(lines_y))
        shuffled_by_class[y] = [lines_y[i] for i in perm]

    out = {}
    for r in ratio_list:
        if r <= 0:
            raise ValueError(f"[ratio-shot] ratio must be > 0, got {r}")

        chosen = []
        for y in classes:
            lines_y = shuffled_by_class[y]
            n_y = len(lines_y)
            if n_y == 0:
                continue

            k = int(np.ceil(r * n_y))
            k = max(int(min_per_class), k)
            k = min(k, n_y)

            # 关键：取前 k 个，而不是重新 random choice
            chosen.extend(lines_y[:k])

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

## within-cross 跨trial做修改（1.27）
def generate_indices_downsample(
    data_dir,
    base_save_dir,
    target_num=40, 
    trial_split_ratio=(0.4, 0.3, 0.3), 
    ratio_list=(0.02, 0.05, 0.1, 0.3, 0.5),
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
    print(f"[Strategy] Structure: Subject -> Label -> Trial -> Segments")
    print(f"[Strategy] Target Samples: {target_num}")


    data_tree = {}
    all_raw_labels = set()
    h5_files = [os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs if f.endswith(".h5")]

    for file_path in tqdm(h5_files, ascii=True, desc="Scanning"):
        try:
            with h5py.File(file_path, "r") as f:
                raw_sid = f.attrs["subject_id"]
                sub_id = str(raw_sid.decode() if isinstance(raw_sid, bytes) else raw_sid)
                if sub_id not in data_tree: data_tree[sub_id] = {}
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and name.endswith("eeg"):
                        l_attr = obj.attrs["label"]
                        lbl = int(l_attr[0] if hasattr(l_attr, '__len__') and not isinstance(l_attr, str) else l_attr)
                        all_raw_labels.add(lbl)

                        if '/' in name:
                            trial_id = name.split('/')[0]
                        else:
                            trial_id = "root_trial" 


                        if lbl not in data_tree[sub_id]: data_tree[sub_id][lbl] = {}
                        if trial_id not in data_tree[sub_id][lbl]: data_tree[sub_id][lbl][trial_id] = []
                        
                        data_tree[sub_id][lbl][trial_id].append(f"{file_path},{name},{lbl}")

                f.visititems(collect)
        except Exception as e:
            print(f"Skipping broken file {file_path}: {e}")

    sorted_labels = sorted(list(all_raw_labels))
    final_map = custom_label_map if custom_label_map else {x: i for i, x in enumerate(sorted_labels)}

    # =========================================================
    # 2. Within (Cross-Trial + Fixed Downsampling)【疾病类数据不做within（只有一个trial）】
    # =========================================================
    processed_data = {}
    flat_list=[]
    for sub_id, label_dict in data_tree.items():
        processed_data[sub_id] = {'train': [], 'val': [], 'test': []}
        
        for raw_lbl, trial_dict in label_dict.items():
            flat_list.append([])
            # print(f"Processing Label {sub_id,raw_lbl, trial_dict}...")
            if raw_lbl not in final_map: continue
            mapped_lbl = final_map[raw_lbl] 
            unique_trials = sorted(list(trial_dict.keys()))
            
            pool_train, pool_val, pool_test = [], [], []

            if len(unique_trials) < 4:
                
                def segment_sort_key(seg_str):
                    # seg_str: "file_path,trial0/segment10/eeg,1"
                    name = seg_str.split(",")[1]              # trial0/segment10/eeg
                    m = re.search(r"segment(\d+)", name)
                    return int(m.group(1)) if m else -1

                # for sub_id_ in sorted(data_tree.keys()):
                # flat_list.append([])
                # for raw_lbl_ in sorted(data_tree[sub_id].keys()):
                for trial_id in sorted(data_tree[sub_id][raw_lbl].keys()):
                    segs = sorted(
                        data_tree[sub_id][raw_lbl][trial_id],
                        key=segment_sort_key
                    )
                    for seg in segs:
                        flat_list[-1].append((sub_id, mapped_lbl, trial_id, seg))


                val_leng_start_id=int(trial_split_ratio[0]*len(flat_list[-1]))
                test_leng_start_id=int((trial_split_ratio[0]+trial_split_ratio[1])*len(flat_list[-1]))
                for seg_id in range(val_leng_start_id):
                    pool_train.append(flat_list[-1][seg_id][-1])
                for seg_id in range(val_leng_start_id, test_leng_start_id):
                    pool_val.append(flat_list[-1][seg_id][-1])
                for seg_id in range(test_leng_start_id, len(flat_list[-1])):
                    pool_test.append(flat_list[-1][seg_id][-1])
            else:
                tr_trials, temp_trials = train_test_split(unique_trials, test_size=(trial_split_ratio[1] + trial_split_ratio[2]), random_state=seed)
                
                if len(temp_trials) > 0:
                    val_ratio_adjusted = trial_split_ratio[1] / (trial_split_ratio[1] + trial_split_ratio[2])
                    va_trials, te_trials = train_test_split(temp_trials, test_size=1.0 - val_ratio_adjusted, random_state=seed)
                else:
                    va_trials, te_trials = [], []
                pool_train = [line for tid in tr_trials for line in trial_dict[tid]]
                pool_val   = [line for tid in va_trials for line in trial_dict[tid]]
                pool_test  = [line for tid in te_trials for line in trial_dict[tid]]

            def sample_fixed(pool, target_n):
                    if not pool: return []
                    cleaned_pool = []
                    for p in pool:
                        parts = p.split(',')
                        cleaned_pool.append(f"{parts[0]},{parts[1]},{mapped_lbl}")
                    
                    n_avail = len(cleaned_pool)
                    n_keep = min(n_avail, target_n)
                    indices = rng.choice(n_avail, size=n_keep, replace=False)
                    return [cleaned_pool[i] for i in indices]

            final_train = sample_fixed(pool_train, 20)
            final_val   = sample_fixed(pool_val,   10)
            final_test  = sample_fixed(pool_test,  10)

            processed_data[sub_id]['train'].extend(final_train)
            processed_data[sub_id]['val'].extend(final_val)
            processed_data[sub_id]['test'].extend(final_test)

    sub_ids = sorted(list(processed_data.keys()))

    print("\n[Output] Saving Within-Subject indices...")
    w_train_all, w_val_all, w_test_all = [], [], []
    
    for sid in sub_ids:
        w_train_all.extend(processed_data[sid]['train'])
        w_val_all.extend(processed_data[sid]['val'])
        w_test_all.extend(processed_data[sid]['test'])
    
    save_list_to_txt(w_train_all, os.path.join(within_save_dir, "train_idx.txt"))
    save_list_to_txt(w_val_all, os.path.join(within_save_dir, "val_idx.txt"))
    save_list_to_txt(w_test_all, os.path.join(within_save_dir, "test_idx.txt"))

    if ratio_list:
        save_ratio_files(w_train_all, within_save_dir, ratio_list, seed, "train_idx_ratio_")

    # ==========================================
        # 3. Cross-Subject Split 
    # ==========================================
    print("\n[Output] Saving Cross-Subject indices...")
    if len(sub_ids) < 3:
        print(f"Warning: Not enough subjects for Cross-Subject split (Found {len(sub_ids)}, need at least 3).")
        print("Skipping Cross-Subject file generation.")
    else:
        try:
            cs_train_subs, temp_subs = train_test_split(sub_ids, test_size=0.6, random_state=seed) 
            if len(temp_subs) < 2:
                 print("Warning: Edge case in subject count. Fallback to manual assignment.")
                 n = len(sub_ids)
                 cs_train_subs = sub_ids[:n-2]
                 cs_val_subs = [sub_ids[n-2]]
                 cs_test_subs = [sub_ids[n-1]]
            else:
                cs_val_subs, cs_test_subs = train_test_split(temp_subs, test_size=0.5, random_state=seed)

            def get_all_samples_from_sub(sid):
                return processed_data[sid]['train'] + processed_data[sid]['val'] + processed_data[sid]['test']

            cs_train_lines = []
            for sid in cs_train_subs: cs_train_lines.extend(get_all_samples_from_sub(sid))
            
            cs_val_lines = []
            for sid in cs_val_subs: cs_val_lines.extend(get_all_samples_from_sub(sid))
            
            cs_test_lines = []
            for sid in cs_test_subs: cs_test_lines.extend(get_all_samples_from_sub(sid))

            save_list_to_txt(cs_train_lines, os.path.join(cross_save_dir, "train_idx.txt"))
            save_list_to_txt(cs_val_lines, os.path.join(cross_save_dir, "val_idx.txt"))
            save_list_to_txt(cs_test_lines, os.path.join(cross_save_dir, "test_idx.txt")) 

            if ratio_list:
                save_ratio_files(cs_train_lines, cross_save_dir, ratio_list, seed, "train_idx_ratio_")

        except Exception as e:
            print(f"Warning: Error during Cross-Subject generation: {e}")
            print("Skipping Cross-Subject generation but keeping Within-Subject files.")
    
    print(cross_save_dir)
    print("[Done] All files generated.")

if __name__ == "__main__":
    data_path = "/zongsheng-group/new_h5/BCIC2A/"
    save_path = "indices"

    generate_indices_downsample(
        data_path,
        save_path,
        target_num=40, 
        trial_split_ratio=(0.6, 0.2, 0.2), 
        ratio_list=(0.05, 0.1, 0.3),
        seed=42,
        custom_label_map=None
    )