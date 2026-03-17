import os
import shutil
from time import time

import numpy as np
import nibabel as nib
import h5py

# Reproducibility
np.random.seed(42)


def z_score_mask_normalize(volume):
    mask = volume > 0
    mean = volume[mask].mean()
    std  = volume[mask].std()
    volume[mask] = (volume[mask] - mean) / std
    return volume

def preprocess_case(case, train_or_test, train_npz_dir, test_h5_dir):
    print(f"Attempting to preprocess [{train_or_test}]: {case}")


    seg = nib.load(f'{RAW_DIR}/{case}/{case}_seg.nii.gz')
    t1ce = nib.load(f'{RAW_DIR}/{case}/{case}_t1ce.nii.gz')
    t2 = nib.load(f'{RAW_DIR}/{case}/{case}_t2.nii.gz')
    flair = nib.load(f'{RAW_DIR}/{case}/{case}_flair.nii.gz')

    seg_data = seg.get_fdata()
    t1ce_data = t1ce.get_fdata()
    t2_data = t2.get_fdata()
    flair_data = flair.get_fdata()

    # Normalize
    t1ce_data = z_score_mask_normalize(t1ce_data)
    t2_data = z_score_mask_normalize(t2_data)
    flair_data = z_score_mask_normalize(flair_data)

    
    # Set seg 4 -> 3 for one hot encoder
    seg_data[seg_data == 4] = 3

    # The shape of the data should have slices at the start
    seg_data = np.transpose(seg_data, (2, 0, 1))
    t1ce_data = np.transpose(t1ce_data, (2, 0, 1))
    t2_data = np.transpose(t2_data, (2, 0, 1))
    flair_data = np.transpose(flair_data, (2, 0, 1))
    # NOW (D, H, W), depth = num slices

    # Test should be an stacked h5 volume 
    if train_or_test == 'test':
        # Stack modalities as channels (3, D, H, W)
        new_name = case + '.npy.h5'

        image_3d = np.stack([t1ce_data, t2_data, flair_data], axis=0)

        hf = h5py.File(os.path.join(test_h5_dir, new_name), 'w')
        hf.create_dataset('image', data=image_3d)   
        hf.create_dataset('label', data=seg_data)  
        hf.close()

    # Test is saved as individual slice,  npz format
    # More memory efficient to slice -> stack
    if train_or_test == 'train':
        num_slices = t2_data.shape[0]

        for s_idx in range(num_slices):
            t1ce_slice = t1ce_data[s_idx,:,:]
            t2_slice = t2_data[s_idx, :, :]
            flair_slice = flair_data[s_idx, :,:]
            seg_slice = seg_data[s_idx, :, :]

            slice_no = "{:03d}".format(s_idx)
            new_name = case +'_slice' + slice_no

            slice_2d = np.stack([t1ce_slice, t2_slice, flair_slice], axis=0)
            np.savez(os.path.join(train_npz_dir, new_name), image=slice_2d, label=seg_slice)

    print(f"Success [{train_or_test}]: {case}")




# Finding all valid data points and writing to lists/brats/valid_subjects.txt
RAW_DIR = './data/brats/BraTS2021_Training_Data'



required_mods = ["flair", "t1", "t1ce", "t2", "seg"]
def has_all_modalities(folder_path, sid):
    for mod in required_mods:
        if not os.path.exists(os.path.join(folder_path, f"{sid}_{mod}.nii.gz")):
            return False
    return True


valid_subjects = []
for d in sorted(os.listdir(RAW_DIR)):
    folder_path = os.path.join(RAW_DIR, d)
    if os.path.isdir(folder_path) and d.startswith("BraTS2021"):
        if has_all_modalities(folder_path, d):
            valid_subjects.append(d)


os.makedirs('./lists/brats/ids', exist_ok=True)
with open('./lists/brats/ids/valid_subjects.txt', 'w') as f:
    f.write('\n'.join(valid_subjects))
####

# Produce Training Split
np.random.shuffle(valid_subjects)
split_idx = int(len(valid_subjects) * 0.8)

train_ids = valid_subjects[:split_idx]
test_ids = valid_subjects[split_idx:]

print("LEN OF TRAIN: ", len(train_ids))
print("LEN OF TEST: ", len(test_ids))


# WRITE DOWN SPLITS
with open('./lists/brats/ids/train_ids.txt', 'w') as f:
    f.write('\n'.join(train_ids))

with open('./lists/brats/ids/test_ids.txt', 'w') as f:
    f.write('\n'.join(test_ids))
##



# 
os.makedirs('./lists/brats/splits', exist_ok=True)


TRAIN_NPZ_DIR = './data/brats/train_npz'
TEST_H5_DIR = './data/brats/test_vol_h5'

os.makedirs(TRAIN_NPZ_DIR, exist_ok=True)
os.makedirs(TEST_H5_DIR, exist_ok=True)


splits = ['train', 'test']
# Now that I have the splits I could just iterate through ids
for split in splits:
    ids = train_ids if split == 'train' else test_ids
    for case in ids:
        preprocess_case(case, split, TRAIN_NPZ_DIR, TEST_H5_DIR)


# NO RESIZING NOW, only on dataloader to match synapse



# Write to list
train_slices = sorted([
    f.replace('.npz', '') 
    for f in os.listdir(TRAIN_NPZ_DIR) 
    if f.endswith('.npz')
])

with open('./lists/brats/splits/train.txt', 'w') as f:
    f.write('\n'.join(train_slices))

print(f"Train slices written: {len(train_slices)}")

with open('./lists/brats/splits/test_vol.txt', 'w') as f:
    f.write('\n'.join(test_ids))

print(f"Test volumes written: {len(test_ids)}")


print(f"Total volumes written: {len(test_ids) + len(train_ids)}")