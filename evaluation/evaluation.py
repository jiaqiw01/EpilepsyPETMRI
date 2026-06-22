from eval_utils import *
from scipy.stats import t
import re, math
import argparse
from datetime import date

all_subj_mask_info = {}
HFILE = h5py.File("xx", 'r')
EXISTING_SUBJECTS = HFILE.keys()

SLICE_START=22 # ignore some noisy slices at the start & end, not used in training
SLICE_END=73 

REFERENCE = 'cerebellum'

RESULTS_DIR = './results'
today = date.today()

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="Where is your test results stored?")
parser.add_argument("--exp", nargs='+', default=[], help="Experiment name")
parser.add_argument("--dest", required=False, help="Where to save evaluation results?", default=RESULTS_DIR)
parser.add_argument("--plot-key", type=str, default='step', help='x-axis')
parser.add_argument("--restart", action='store_true', default=True)
parser.add_argument("--external", action='store_true', default=False)
parser.add_argument("--suffix",  default='')

parser.add_argument("--plot", action='store_true', default=True)
parser.add_argument("--filter", action='append')
parser.add_argument("--val", action='append')

def filter_subfolders(folders, by=['step'], val=['100']):
    assert len(by) == len(val), "Please give paired criterion"
    filtered_folders = None
    for i, k in enumerate(by):
        v = val[i]
        # get corresponding folders
        pattern = fr'{k}_?{v}(?!\d)'
        found = [m for f in folders for m in re.findall(pattern, f)]
        if not filtered_folders:
            filtered_folders = found
        else:
            filtered_folders = np.intersect1d(filtered_folders, found)
    print("Filtered subfolders: ", filtered_folders)
    return set(filtered_folders)


def run_congruency_measure(test_subj_tracer_info, all_subj_mask_info, exp_type, plot=False):
    if 'fdg - tumor' in test_subj_tracer_info and 'fdg' in test_subj_tracer_info:
        fdg_total = test_subj_tracer_info['fdg - tumor'] + test_subj_tracer_info['fdg']
    else:
        fdg_total = test_subj_tracer_info['fdg']

    all_metrics = {}
    fig1, axes = plt.subplots(nrows=4, ncols=len(fdg_total)//4 + 1, figsize=(15,25))
    axes_flatten_1 = axes.ravel()
    cong_idx_all = []
    cong_mae_all = []
    info = {}
    for i, subj in enumerate(sorted(fdg_total)):
        print(f"====> Running congruency for subject {subj}")
        content = all_subj_mask_info[subj]
        roi_masks_left_right = content['mask_left_right']
        roi_mask_global = content['mask_global']
        area_info = content['area_info']
        pets = content['pets']
        suvr_asym, suvr_asym_diff, cong_index, cong_mae = compute_asymmetry(roi_masks_left_right, roi_mask_global, pets, area_info)
        print(subj + ": " + str(cong_index[exp_type]) + " " + str(cong_mae[exp_type]))
        info[subj] = {
            'ci': cong_index[exp_type],
            'cmae': cong_mae[exp_type]
        }
        # cong_idx_all += cong_index[exp_type]
        cong_idx_all.append(cong_index[exp_type])
        cong_mae_all.append(cong_mae[exp_type])
        if 'suvr_asym' not in all_metrics:
            all_metrics['suvr_asym'] = []
        if 'suvr_asym_diff' not in all_metrics:
            all_metrics['suvr_asym_diff'] = []
        all_metrics['suvr_asym'].append(suvr_asym)
        all_metrics['suvr_asym_diff'].append(suvr_asym_diff)
        if plot:
            plot_asymmetry(suvr_asym, subj, axes_flatten_1[i])

    cong_idx_mean = np.mean(cong_idx_all)
    cong_mae_mean = np.mean(cong_mae_all)
    std_ci = np.std(cong_idx_all, ddof=1) / np.sqrt(len(cong_mae_all))
    std_err = np.std(cong_mae_all, ddof=1) / np.sqrt(len(cong_mae_all))  # Standard error
    # print(cong_mae_mean, std_err)
    # # Compute 95% confidence interval using t-distribution
    confidence_interval_cmae = t.interval(0.95, len(cong_mae_all)-1, loc=cong_mae_mean, scale=std_err)
    confidence_interval_ci = t.interval(0.95, len(cong_idx_all)-1, loc=cong_idx_mean, scale=std_ci)

    print(f"======> Average CMAE for {exp_type}: {cong_mae_mean}, CI: {confidence_interval_cmae} <=====")
    print(f"======> Average CI for {exp_type}: {cong_idx_mean}, CI: {confidence_interval_ci} <=====")
    return {
        'CMAE Mean': cong_mae_mean,
        'CMAE CI': confidence_interval_cmae,
        'CI Mean': cong_idx_mean,
        'CI CI': confidence_interval_ci
    }

def run_pixelwise_SUVR(subjects, test_subj_tracer_info, all_subj_mask_info, exp_type, relative_diff=False):
    all_synth_dsuvr_mean_fdg = []
    all_synth_dsuvr_mean_amy = []
    all_synth_dsuvr_std_fdg = []
    all_synth_dsuvr_std_amy = []

    stats_dict = {
        # 'fdg - tumor': [],
        'fdg': [],
        'amyvid': []
    }

    for subj in subjects:
        seg_file = f"xx/{subj}/{TARGET_SEG_FILE}"
        # seg = nib.load(seg_file).get_fdata()[:, :, :]
        # mask = seg != 0
        print(f"===> Running SUVR for subject {subj}")
        content = all_subj_mask_info[subj]
        normed_truth = content['pets']['truth']
        normed_synth = content['pets'][exp_type]

        roi_mask_global = content['mask_global']
        brain_volume_truth = normed_truth
        brain_volume_zero = normed_synth


        # roi_masks_global, area_info = compute_masks(seg_file)
        ref_mask = roi_mask_global[REFERENCE]
        
        # print(brain_volume_truth.min(), brain_volume_truth.max())
        true_suv_ref = (brain_volume_truth * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
        true_suvr_volume = brain_volume_truth / true_suv_ref

        synth_suv_ref = (brain_volume_zero * ref_mask).sum() / ref_mask.sum() # cerebellum total suv / cerebellum total volume
        synth_suvr_volume = brain_volume_zero / synth_suv_ref

        synth_mean = np.abs(true_suvr_volume - synth_suvr_volume).mean()
        synth_std = np.abs(true_suvr_volume - synth_suvr_volume).std()

        if 'fdg - tumor' in test_subj_tracer_info and subj in test_subj_tracer_info['fdg - tumor']:
            stats_dict['fdg - tumor'].append([synth_mean, synth_std])
            print(f"====> FDG Tumor {subj} SUVR Mean: {synth_mean}, SUVR STD: {synth_std}")
            all_synth_dsuvr_mean_fdg.append(synth_mean)
            all_synth_dsuvr_std_fdg.append(synth_std)

        elif subj in test_subj_tracer_info['fdg']:
            stats_dict['fdg'].append([synth_mean, synth_std])
            print(f"====> FDG {subj} SUVR Mean: {synth_mean}, SUVR STD: {synth_std}")
            all_synth_dsuvr_mean_fdg.append(synth_mean)
            all_synth_dsuvr_std_fdg.append(synth_std)


    mean_of_mean_fdg = np.array(all_synth_dsuvr_mean_fdg).mean()
    mean_of_std_fdg = np.array(all_synth_dsuvr_std_fdg).mean()

    # ----FDG----

    std_err_mean_fdg = np.std(all_synth_dsuvr_mean_fdg, ddof=1) / np.sqrt(len(all_synth_dsuvr_mean_fdg))  # Standard error
    ci_mean_fdg = t.interval(0.95, len(all_synth_dsuvr_mean_fdg)-1, loc=mean_of_mean_fdg, scale=std_err_mean_fdg) # # Compute 95% confidence interval using t-distribution

    std_err_std_fdg = np.std(all_synth_dsuvr_std_fdg, ddof=1) / np.sqrt(len(all_synth_dsuvr_std_fdg))  # Standard error
    ci_std_fdg = t.interval(0.95, len(all_synth_dsuvr_std_fdg)-1, loc=mean_of_std_fdg, scale=std_err_std_fdg) # # Compute 95% confidence interval using t-distribution

    print(f"=====> Finished delta_SUVR for {exp_type} <====== ")
    print(f"=====> FDG: Mean = {mean_of_mean_fdg}, CI = {ci_mean_fdg}, Mean STD = {mean_of_std_fdg}, CI = {ci_std_fdg}")

    return {
        'FDG': {
            'Mean Mean': mean_of_mean_fdg,
            'Mean CI': ci_mean_fdg,
            'Mean STD': mean_of_std_fdg,
            'Mean STD CI': ci_std_fdg
        },
    }



def run_eval_basic(sub_folders, src_root, save_root, plot=True):
    all_exp_result_dict = {}
    for exp_type in sub_folders:
        # create result folders
        save_dest = os.path.join(save_root, exp_type)
        os.makedirs(save_dest, exist_ok=True)

        # compute masks and stuff
        test_dir = os.path.join(src_root, exp_type) # place to load test files
        test_subjects = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
        valid_subjects = []
        test_subj_tracer_info = {
            'fdg - tumor' : [],
            'fdg': [],
        }

        for i, subj in enumerate(test_subjects):
            if subj in all_subj_mask_info.keys() and 'mask_global' in all_subj_mask_info[subj].keys() and 'area_info' in all_subj_mask_info[subj].keys():
                pets = gather_files(subj, test_dir, exp_type, TARGET_ROIS, TARGET_LR_ROIS, TARGET_SEG_FILE, pets_only=True, slice_start=SLICE_START, slice_end=SLICE_END)
                all_subj_mask_info[subj]['pets'] = pets
                print(f"Subject {subj} already found")
            else:
                try: # not all of them have valid segmentations though...
                    pets, roi_mask_global, roi_mask_left_right, area_info = gather_files(subj, test_dir, exp_type, TARGET_ROIS, TARGET_LR_ROIS, TARGET_SEG_FILE, slice_start=SLICE_START, slice_end=SLICE_END)
                    all_subj_mask_info[subj] = {'pets': pets,
                                            'mask_global': roi_mask_global,
                                            'mask_left_right': roi_mask_left_right,
                                            'area_info': area_info}
                except:
                    print(f"===> {subj} skipped, probably due to invalid segmentation")
                    continue
            if subj in EXISTING_SUBJECTS:
                att = HFILE[subj].attrs['label']
                all_subj_mask_info[subj]['label'] = att
                if 'tumor' in att:
                    test_subj_tracer_info['fdg - tumor'].append(subj)
                elif 'fdg' in att:
                    test_subj_tracer_info['fdg'].append(subj)
            if subj not in valid_subjects:
                valid_subjects.append(subj)
                print(f"====> {subj} added to evaluation <====")

        print("====== Evaluating Congruency Metrics ======")
        print("====== SUVR Asymmetry is computed using (left - right) / (left + right) ======")
        result_cong = run_congruency_measure(test_subj_tracer_info, all_subj_mask_info, exp_type, plot=plot)
        # run SUVRs
        print("====== Evaluating SUVR Metrics ======")
        result_suvr = run_pixelwise_SUVR(valid_subjects, test_subj_tracer_info, all_subj_mask_info, exp_type)

        all_exp_result_dict[exp_type] = {
            'Congruence': result_cong,
            'SUVR': result_suvr,
            # 'sampling efficiency': sampling_time
        }
        # Save visualizations
        if plot:
            print("Saving Slice Visualizations....")
            plot_slice_visulization_all_subj(valid_subjects, all_subj_mask_info, exp_type, save=os.path.join(save_dest, 'slices.png'))
            for subj in valid_subjects:
                plot_slice_visualization_one_subject(subj, all_subj_mask_info, exp_type, save=os.path.join(save_dest, f'subject_{subj}_slices.png'))
    return all_exp_result_dict


def plot_metrics(plot_key, metric_file, metrics_to_plot=['CI', 'CMAE', 'SUVR Mean', 'SUVR STD', 'sampling efficiency'], tracer_to_plot='FDG'):
    with open(metric_file, 'r') as fr:
        res = json.load(fr)

    Xs = sorted([int(x) for x in res.keys()])
    print(Xs)
    all_metrics = {}
    ymin = None
    ymax = None
    for metric in metrics_to_plot:
        if metric == 'CI' or metric == 'CI Mean':
            vals = np.array([res[str(x)]['Congruence']['CI Mean'] for x in Xs])
        elif metric == 'CMAE' or metric == 'CMAE Mean':
            vals = np.array([res[str(x)]['Congruence']['CMAE Mean'] for x in Xs], dtype=float) * 1000
        elif metric == 'SUVR Mean':
            vals = np.array([res[str(x)]['SUVR'][tracer_to_plot]['Mean Mean'] for x in Xs], dtype=float) * 100
        elif metric == 'SUVR STD':
            vals = np.array([res[str(x)]['SUVR'][tracer_to_plot]['Mean STD'] for x in Xs], dtype=float) * 100
        elif metric == 'sampling efficiency':
            vals = np.ceil(np.array([res[str(x)]['sampling efficiency'] for x in Xs]) / 60)
        else:
            raise NotImplementedError
        all_metrics[metric] = vals
        if not ymin:
            ymin = vals.min()
        if not ymax:
            ymax = vals.max()
        ymin = min(ymin, vals.min())
        ymax = max(ymax, vals.max())
    
    plt.clf()
    n_metrics = len(all_metrics)
    n_categories = len(Xs)

    x = np.arange(n_categories)  # the label locations
    bar_width = 0.8 / n_metrics  # total width of all bars per group (up to 1.0)

    # Plot bars
    fig, ax = plt.subplots()
    for i, (metric_name, metric_values) in enumerate(all_metrics.items()):
        offset = (i - n_metrics / 2) * bar_width + bar_width / 2
        ax.bar(x + offset, metric_values, width=bar_width, label=metric_name)
    # Labels & formatting
    ax.set_xlabel(plot_key)
    ax.set_ylabel('Vals')
    ax.set_title(f'Performance Metrics by {plot_key}')
    ax.set_xticks(x)
    ax.set_xticklabels(Xs)
    ax.set_ylim(ymin, ymax)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(results_subdir, f'Metrics_plot_by_{plot_key}.png'), dpi=300, bbox_inches='tight')
    


if __name__ == '__main__':
    args = parser.parse_args()
    src_root = args.src
    dest_root = args.dest
    SUB_FOLDERS = os.listdir(src_root)
    plot = True if args.plot else False
    # EXP = args.exp
    if args.exp != []:
        SUB_FOLDERS = args.exp
    results_subdir = os.path.join(args.dest, f'{str(today)}')
    os.makedirs(results_subdir, exist_ok=True)
    if args.restart:
        res = run_eval_basic(SUB_FOLDERS, src_root, dest_root, plot=plot)  
        if args.exp != []:
            exp_name = '_'.join(args.exp)
            fname = os.path.join(results_subdir, f'metrics_{exp_name}_{args.suffix}.json')
        else:
            fname = os.path.join(results_subdir, f'metrics_all.json')
        with open(fname, 'w') as fw:
            json.dump(res, fw, indent=4)
            print(f"Saved in {os.path.join(results_subdir, 'metrics.json')}")
