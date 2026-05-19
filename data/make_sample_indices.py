import os
import h5py
import numpy as np
from tqdm import tqdm

def save_list_to_txt(data_list, file_path):
    """高效保存列表到txt"""
    if not data_list: return
    with open(file_path, "w") as f:
        f.write("\n".join(data_list) + "\n")

def generate_indices_fixed_num(
    data_dir,
    base_save_dir,
    split_ratio=(0.4, 0.3, 0.3),  # Train, Val, Test
    num_list=(10,20,40, 50,70, 80, 100, 120),  # 每个Subject每个Label保留的总样本数
    seed=42,
    custom_label_map=None
):
    rng = np.random.RandomState(seed)
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    save_dir = os.path.join(base_save_dir, f"{dataset_name}_indices_within_subject_downsample")
    os.makedirs(save_dir, exist_ok=True)

    # 1. 快速扫描与分组
    # data_pool[sub_id][label] = [line, line, ...]
    data_pool = {}
    h5_files = [os.path.join(r, f) for r, _, fs in os.walk(data_dir) for f in fs if f.endswith(".h5")]
    
    print(f"Scanning {len(h5_files)} files...")
    all_labels = set()

    for file_path in tqdm(h5_files, ascii=True):
        try:
            with h5py.File(file_path, "r") as f:
                sub_id = str(f.attrs["subject_id"].decode() if isinstance(f.attrs["subject_id"], bytes) else f.attrs["subject_id"])
                if sub_id not in data_pool: data_pool[sub_id] = {}

                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset) and name.endswith("eeg"):
                        lbl = int(obj.attrs["label"][0] if hasattr(obj.attrs["label"], '__len__') and not isinstance(obj.attrs["label"], str) else obj.attrs["label"])
                        if lbl not in data_pool[sub_id]: data_pool[sub_id][lbl] = []
                        
                        # 格式: path,name,label (后续处理统一Map)
                        data_pool[sub_id][lbl].append((file_path, name, lbl))
                        all_labels.add(lbl)
                f.visititems(collect)
        except Exception as e:
            print(f"Skip {file_path}: {e}")

    # 2. Label Map 处理
    sorted_labels = sorted(list(all_labels))
    final_map = custom_label_map if custom_label_map else {x: i for i, x in enumerate(sorted_labels)}
    print(f"Unique Labels: {sorted_labels} -> Map: {final_map}")

    # 3. 转换为最终字符串列表，并计算最小样本数（用于快速判断）
    processed_pool = {} # {sub: {mapped_label: [str_line, ...]}}
    min_samples_global = float('inf')

    for sub, lab_dict in data_pool.items():
        processed_pool[sub] = {}
        for raw_lbl, items in lab_dict.items():
            if raw_lbl not in final_map: continue
            mapped_lbl = final_map[raw_lbl]
            
            lines = [f"{fp},{fn},{mapped_lbl}" for (fp, fn, _) in items]
            processed_pool[sub][mapped_lbl] = lines
            
            if len(lines) < min_samples_global:
                min_samples_global = len(lines)

    sub_ids = sorted(processed_pool.keys())
    

    print("Generating Full Dataset indices...")
    full_train, full_val, full_test = [], [], []
    for sub in sub_ids:
        for lbl in final_map.values():
            lines = processed_pool[sub].get(lbl, [])
            if not lines: continue
            
            # 全量打乱划分
            curr_lines = np.array(lines)
            rng.shuffle(curr_lines)
            
            n = len(curr_lines)
            n_tr = int(n * split_ratio[0])
            n_va = int(n * split_ratio[1])
            
            full_train.extend(curr_lines[:n_tr])
            full_val.extend(curr_lines[n_tr:n_tr+n_va])
            full_test.extend(curr_lines[n_tr+n_va:])

    save_list_to_txt(full_train, os.path.join(save_dir, "train_idx.txt"))
    save_list_to_txt(full_val, os.path.join(save_dir, "val_idx.txt"))
    save_list_to_txt(full_test, os.path.join(save_dir, "test_idx.txt"))

    sorted_nums = sorted([int(x) for x in num_list])
    
    for n in sorted_nums:
        
        print(f"Generating indices for Target N={n} (cap at max available)...")
        
        curr_train, curr_val, curr_test = [], [], []
        
        for sub in sub_ids:
            for lbl in final_map.values():
                lines = processed_pool[sub].get(lbl, [])
                total_avail = len(lines)
                
                if total_avail == 0:
                    continue
                actual_k = min(n, total_avail)
                
                chosen = rng.choice(lines, size=actual_k, replace=False)
                
                n_tr = int(actual_k * split_ratio[0])
                n_va = int(actual_k * split_ratio[1])
                
                curr_train.extend(chosen[:n_tr])
                curr_val.extend(chosen[n_tr : n_tr + n_va])
                curr_test.extend(chosen[n_tr + n_va :])

        # 保存文件
        suffix = f"num_{n}"
        save_list_to_txt(curr_train, os.path.join(save_dir, f"train_idx_{suffix}_{seed}.txt"))
        save_list_to_txt(curr_val, os.path.join(save_dir, f"val_idx_{suffix}_{seed}.txt"))
        save_list_to_txt(curr_test, os.path.join(save_dir, f"test_idx_{suffix}_{seed}.txt"))

    print(f"Success. Files saved to: {save_dir}")

if __name__ == "__main__":
    # 使用示例
    data_path = "/zongsheng-group/ISRUC-Sleep_1/"
    save_path = "indices" 
    
    generate_indices_fixed_num(
        data_dir=data_path,
        base_save_dir=save_path,
        split_ratio=(0.4, 0.3, 0.3),     # 划分比例
        num_list=(10, 20, 40, 60, 80, 100,120), # 这里的数字是 (Train+Val+Test) 的总和
        seed=42
    )