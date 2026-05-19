import os
import numpy as np
import torch
import mne

# USER_DEFINED_CHANNELS = [
#     'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
#     'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
#     'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
#     'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
#     'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
#     'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
#     'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
#     'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
#     'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
#     'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
#     'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
#     'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h'
# ]

SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}

def normalize_pos(pos: np.ndarray, eeg_mask):
    if eeg_mask.any():
        eeg_mean = np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] -= eeg_mean
        eeg_scale = np.sqrt(3 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
        if eeg_scale < 1e-6:
            eeg_scale = 1.0 
        pos[eeg_mask, :3] /= eeg_scale
    return pos

def generate_all_possible_pos_info():

    montage = mne.channels.make_standard_montage('standard_1005')
 
    all_ch_names = montage.ch_names
    positions_3d = montage.get_positions()['ch_pos']

    pos_lookup = {k.upper(): v for k, v in positions_3d.items()}

    final_pos_list = []
    final_type_list = []
    valid_names = []
    found_count = 0

    for name in all_ch_names:
        u_name = name.upper()
        

        if u_name in pos_lookup and pos_lookup[u_name] is not None:
            xyz = pos_lookup[u_name]
            final_pos_list.append(np.hstack([xyz, [0.0, 0.0, 0.0]]))
            final_type_list.append(SENSOR_TYPE_DICT["EEG"])
            valid_names.append(name)
            found_count += 1
        else:

            continue


    pos_array = np.array(final_pos_list, dtype=np.float32)
    type_array = np.array(final_type_list, dtype=np.int32)

    eeg_mask = (type_array == SENSOR_TYPE_DICT["EEG"])
    pos_array = normalize_pos(pos_array, eeg_mask)
    
    print(f"提取完成。共计保存 {found_count} 个带有标准坐标的通道。")
    return pos_array, type_array, valid_names

if __name__ == "__main__":
    pos, sensor_type, all_names = generate_all_possible_pos_info()
    
    save_path = "data/standard_coords_all.pt"
    os.makedirs("data", exist_ok=True)
    torch.save({
        "ch_names": all_names,         
        "pos": torch.from_numpy(pos),
        "sensor_type": torch.from_numpy(sensor_type)
    }, save_path)
    print(f"Saved {len(all_names)} channels to: {save_path}")