import argparse
import h5py
import os
import numpy as np
import json
import logging
from datetime import datetime

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename="./logs/train_test_split.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does')
parser.add_argument('-s', '--src', type=str, help='Path to the source H5 file containing the data')
parser.add_argument('-c', '--contrast', nargs='+', help='List of contrasts to include in the split')
parser.add_argument('-r', '--restart', action="store_true", help="Whether restart the generation or continue")
parser.add_argument('-d', '--dest', required=True, help="where to store")

args = parser.parse_args()

def filter_data_by_contrasts_and_tracer(data_h5, contrast_list=['T1', 'T2_FLAIR', 'PET']):
    filter_dict = {}
    valid_subjects = []
    skipped_subjects = []
    for c in contrast_list:
        filter_dict[c] = False

    with h5py.File(data_h5, "r") as f:
        all_subjects = list(f.keys())
        for sub in all_subjects:

            subject_data = f[sub]
            found = filter_dict.copy()

            for req in filter_dict.keys():
                for k in subject_data:
                    if 't1' in req.lower():
                        if 'gre' in k.lower() or 'se' in k.lower():
                            found[req] = True
                    elif 't1c' in req.lower():
                        if k == 'T1c_GRE' or k == 'T1c_GRE':
                            found[req] = True
                    elif 'flair' in req.lower():
                        if req == k:
                            found[req] = True
                    else: # pet can have multiple names
                        if req in k:
                            found[req] = True

            valid = True
            for _ in found.keys():
                if found[_] == False:
                    logging.info(f"No {_} found for subject {sub}.... Skipping it")
                    valid = False
                    break

            if valid: 
                valid_subjects.append(sub)
            else:
                skipped_subjects.append(sub)
    logging.info(f"====> Found {len(valid_subjects)} valid subjects by tracer and contrasts!")
    return valid_subjects, skipped_subjects


def generate_train_val_test_slices(valid_subjects, dest, new_start=False, num_test=10, split=0.9):

    all_val_only_subj = []
    all_train_only_subj = []

    meta = {}
    if new_start:
        np.random.shuffle(valid_subjects) 

    # for each tracer, collect train/val/test subjects
    # print(new_subjects, old_subjects)
    test_subj = np.random.choice(valid_subjects, num_test, replace=False).tolist()

    test_subj = np.array(test_subj)
    train_val_subj = np.setdiff1d(valid_subjects, test_subj)
    logging.info(f"===> Collected test subjects: {test_subj}")

    meta = {
        'train_val': train_val_subj.tolist(),
        'test': test_subj.tolist(),
        'num_train_val': len(train_val_subj),
        'num_test': len(test_subj)
    }
    

    slice_file_train = os.path.join(dest, f'train.txt')
    slice_file_val = os.path.join(dest, f'val.txt')
    test_file = os.path.join(dest, 'test_subj.txt')
   
    if new_start:
        mode = 'w'
    else:
        mode = 'a'

    split_idx = int((1-split) * len(train_val_subj)) # first part is training, second part val

    with open(slice_file_train, mode) as fw2, open(slice_file_val, mode) as fw3:
        for i, subj in enumerate(train_val_subj):
            slice_range = [18, 71]
            for j in range(slice_range[0], slice_range[1]):
                if i < split_idx:
                    fw3.write(subj+' '+str(j)+'\n')
                else:
                    fw2.write(subj+' '+str(j)+'\n')
            if i < split_idx:
                all_val_only_subj.append(subj)
            else:
                all_train_only_subj.append(subj)
        # write test subjects
        with open(test_file, mode) as fw:
            for t in test_subj:
                fw.write(f"{t}\n")
            
    return all_train_only_subj, all_val_only_subj, test_subj, meta


def train_split_randomized(dest, h5,  required_contrast=['T1', 'T2_FLAIR', 'PET'], new_start=False, suffix=''):
    name = '_'.join(required_contrast)
    name = name + '_' + suffix

    dest = os.path.join(dest, name)
    os.makedirs(dest, exist_ok=True)
    logging.info(f"=====Saving in {dest}=====")
    logging.info(f"=====Using Contrast: {required_contrast} =====")
    if new_start:
        logging.info("===> Restart...")
    # Test train splits

    valid_subjects, skipped = filter_data_by_contrasts_and_tracer(h5, required_contrast)
    # print("Valid subjects....", valid_subjects)
    train_subj, val_subj, test_subj, meta_info = generate_train_val_test_slices(valid_subjects, dest, new_start=new_start)

    if not new_start and os.path.exists(os.path.join(dest, 'info.json')):
        with open(os.path.join(dest, 'info.json'), 'r') as fr:
            info = json.load(fr)
            info['num_valid_subj'] = len(valid_subjects)
            info['num_train_subj'] = len(train_subj)
            info['num_val_subj'] = len(np.unique(val_subj).tolist()),
            info['num_test_subj'] = len(test_subj)
            info['valid_subjects'] = valid_subjects
            info['train_subj'] = np.unique(train_subj).tolist()
            info['val_subj'] = np.unique(val_subj).tolist()
            info['test_subj'] = np.unique(test_subj).tolist()
    else:
        logging.info("=================")
        logging.info(f"{len(train_subj)}, {len(val_subj)}")
        info = {
            'num_train_subj': len(train_subj),
            'num_val_subj': len(val_subj),
            'num_test_subj': len(test_subj),
            'num_skipped_subj': len(skipped),
            'valid_subjects': valid_subjects,
            'all_train_subj': train_subj,
            'all_val_subj': val_subj,
            'all_test_subj': test_subj.tolist(),
            'skipped_subjects': skipped,
            'details': meta_info
        }

    with open(os.path.join(dest, 'info.json'), 'w') as fw:
        json.dump(info, fw, indent=4)
    logging.info("===> All Done! <===")

if __name__ == "__main__":
    restart = True if args.restart else False
    
    dest = args.dest
    h5 = args.src # /data/jiaqiw01/PET_MRI/data/all_cases_rawspace_complete_max.h5
    required_contrast = args.contrast

    suffix = datetime.now().strftime("%Y%m%d")
    
    train_split_randomized(dest, h5, required_contrast=required_contrast, new_start=restart, suffix=suffix)
