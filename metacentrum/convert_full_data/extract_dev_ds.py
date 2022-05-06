# %%
import os
import numpy as np
import pandas as pd

from natsort import natsorted

import tss

# %%
run_folder = '../run-datafiles/'
run_files = os.listdir(run_folder)
run_files = natsorted(run_files)

rf = run_files[0]
npz = np.load(run_folder + rf)

list(npz.keys())
# %%
kernel_names = ['lin', 'quad', 'se', 'matern5', 'rq', 'nn-arcsin', 'add', 'se+quad', 'gibbs']

fun, dim, ker, ins = rf.split('.')[0].split('_')[-4:]
dim = dim[:-1]

ds_dict = {}
for file in run_files:
    fun, dim, rid, ins = file.split('.')[0].split('_')[-4:]
    fun = int(fun)
    dim = int(dim[:-1])
    ker = (int(rid) - 1) % 9
    ins = int(ins)

    dct = ds_dict
    for key in [dim, fun, ker]:
        dct = dct.setdefault(key, {})

    dct[ins] = run_folder + file


# %%
def condition_num(matrix):
    # vectors in rows
    vec_lengths = np.linalg.norm(matrix, axis=1)
    return vec_lengths.max() / vec_lengths.min()


def fst_scnd_ratio(matrix):
    # vectors in rows
    vec_lengths = np.linalg.norm(matrix, axis=1)
    first, second, *_ = np.flip(np.sort(vec_lengths))
    return first / second


# %%
class RecordedRun:
    def __init__(self, npz_path, info):
        super().__init__()

        npz = np.load(npz_path)

        self.dim = npz['dimensions']
        self.fun = npz['function_id']

        self.xmeans = npz['surrogate_data_means'].T
        self.sigmas = npz['surrogate_data_sigmas']
        self.bds = npz['surrogate_data_bds']
        self.iruns = npz['iruns']
        self.evals = npz['evals']
        self.points = npz['points']
        self.fvalues = npz['fvalues']
        self.orig = npz['orig_evaled']
        self.coco = npz['fvalues_orig']
        self.gen_split = npz['gen_split']

        self.n_gen = len(self.gen_split)
        self.n_points = len(self.points)

        # tuple: x_train, y_train, x_test, y_test, tss_mask, stats
        self.all_gens = [self.get_gen(gen_i, info) for gen_i in range(1, self.n_gen)]
        self.stats_table = pd.DataFrame([gen[-1] for gen in self.all_gens])

        # for key, val in info.items():
        #     self.stats_table[key] = val
        # self.stats_table['ins'] = ins

    def get_gen(self, gen_i, info):
        # first point is initial guess
        # gen_split[0] = 0, gen_split[1] = 1, gen_split[2] = 1, ...
        low = self.gen_split[gen_i] + 1
        high = self.gen_split[gen_i + 1] + 1 if gen_i + 1 < self.n_gen else self.n_points

        x_test = self.points[low:high]
        y_test = self.coco[low:high]

        o = self.orig[:low]
        x_train = self.points[:low][o]
        y_train = self.coco[:low][o]

        pop = x_test

        mean = self.xmeans[gen_i]
        sigma = self.sigmas[gen_i]
        bd = self.bds[gen_i]
        mahalanobis_transf = np.linalg.inv(bd * sigma)

        maximum_distance = 4  # trainRange
        maximum_number = int(20 * dim)

        tss2 = tss.TSS2(pop, mahalanobis_transf, maximum_distance, maximum_number)
        tss_mask, _ = tss2(x_train, y_train)

        x_tss = x_train[tss_mask]
        y_tss = y_train[tss_mask]

        stats = {
            "gen_num": gen_i,
            "restarts": self.iruns[gen_i] - 1,
            "arch_len": len(x_train),
            "tss_len": np.count_nonzero(tss_mask),
            "var_y_tss": np.var(y_tss),

            "sigma": sigma,
            "bd": bd,
            "mean": mean,
            "cond_num": condition_num(bd.T),
            "fst_scnd_ratio": fst_scnd_ratio(bd.T),
            "max_eigenvec": max(np.linalg.norm(bd.T, axis=1))
        }
        stats.update(info)

        return x_train, y_train, x_test, y_test, tss_mask, stats


# %%
def mark_selected_evenly(table, col_name, num_to_select):
    col = table[col_name]
    col_w_nums = np.array([col, np.arange(len(col))]).T
    col_non_nan = col_w_nums[~np.isnan(col_w_nums[:, 0]), :]
    k_even_spread = np.linspace(0, len(col_non_nan) - 1, num=num_to_select, dtype=int)

    selected = col_non_nan[col_non_nan[:, 0].argsort(),1][k_even_spread].astype(int)

    for row_i in selected:
        if not table.at[row_i, 'selected']:
            table.at[row_i, 'selected'] = True
        else:
            free = np.where(~table['selected'])[0]
            if len(free) == 0:
                break
            closest = free[np.abs(free - row_i).argmin()]
            table.at[closest, 'selected'] = True


def select_from_all_instances(case, num_to_select=5):
    runs = []
    for ins, npz_path in case['instances']:
        info = case.copy()
        del info['instances']
        info['ins'] = ins

        runs.append(RecordedRun(npz_path, info))

    all_ins_table = pd.concat([rr.stats_table for rr in runs], ignore_index=True)

    all_ins_table['selected'] = False
    mark_selected_evenly(all_ins_table, 'var_y_tss', num_to_select)
    mark_selected_evenly(all_ins_table, 'sigma', num_to_select)
    mark_selected_evenly(all_ins_table, 'gen_num', num_to_select)
    mark_selected_evenly(all_ins_table, 'cond_num', num_to_select)
    mark_selected_evenly(all_ins_table, 'max_eigenvec', num_to_select)

    selected_indexes = np.where(all_ins_table['selected'])
    all_ins_data = [rr.all_gens for rr in runs if len(rr.all_gens) > 0]
    try:
        selected_data = np.concatenate(all_ins_data, dtype=object)[selected_indexes]
    except:
        pass
    return selected_data


# %%
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

for dim, dct in ds_dict.items():
    if dim != 20:
        continue

    dev_ds_filename = 'dev_ds_dim{}'.format(dim)

    try:
        loaded = np.load(dev_ds_filename + '.npy', allow_pickle=True)
    except FileNotFoundError:
        loaded = []

    to_skip = set()
    for case in loaded:
        ker = case[-1]['ker']
        fun = case[-1]['fun']
        to_skip.add((fun, ker))

    ds_list_by_dim = []
    for fun, dct in dct.items():
        for ker, dct in dct.items():
            instances = list(dct.values())
            kernel = kernel_names[ker]
            if kernel not in ['se', 'matern5', 'rq']:
                continue

            if (fun, kernel) in to_skip:
                print("Already done - FUN: {}, DIM: {}, KER: {}".format(fun, dim, kernel))
                continue

            run_dict = {
                'fun': fun,
                'dim': dim,
                'ker': kernel,
                'instances': list(dct.items())
            }

            selected = select_from_all_instances(run_dict)

            ds_list_by_dim.append(selected)
            print("Selected from runs: FUN: {}, DIM: {}, KER: {}".format(fun, dim, kernel))
            in_this_run = np.concatenate(ds_list_by_dim)
            file_content = np.concatenate([loaded, in_this_run]) if len(loaded) > 0 else in_this_run
            np.save(dev_ds_filename, file_content)
            print("Saved {} records to {}".format(len(ds_list_by_dim), dev_ds_filename))
