import pandas as pd
from tqdm import tqdm
import numpy as np

def model_to_col(model_name_lis, topk, topk_tmp):
    cols = []
    model_idx = []
    if 'git' in model_name_lis:
        model_idx.append(0)
    if 'beit3' in model_name_lis:
        model_idx.append(1)
    if 'ofa' in model_name_lis:
        model_idx.append(2)
    if 'blip2_large' in model_name_lis:
        model_idx.append(3)
    for idx in model_idx:
        for i in range(topk):
            cols.append(idx * topk_tmp + (topk_tmp-topk) + i)
    return np.array(cols)

def view_candidates(candidate_idx_fn, candidate_fn, model_lis, topk, add_model):
    if add_model:
        model_res = []
        for model in model_lis:
            df = pd.read_csv(f'datasets/nice/model_output/{model}.csv')
            model_res_single = df.loc[:, 'caption'].tolist()
            model_res.append(model_res_single)
    in_df = pd.read_csv(candidate_idx_fn)
    selected_cols = []
    model_num = len(model_lis)
    selected_cols = model_to_col(model_lis, topk, 5)
    print(selected_cols)
    in_df = in_df.to_numpy()
    in_df = in_df[:, selected_cols]
    candidate_caps = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    cols = []
    for model in model_lis:
        for i in range(topk):
            cols.append(model)
    if add_model:
        for model in model_lis:
            cols.append(model)
    out_df = pd.DataFrame(columns=cols, index=range(20000))
    for i in tqdm(range(20000)):
        for j in range(model_num * topk):
            idx = in_df[i, j]
            cap = candidate_caps.iloc[i, idx+1]
            out_df.iloc[i, j] = cap
    if add_model:
        for k in range(model_num):
            out_df.iloc[:, model_num * topk + k] = model_res[k]
    out_df.to_csv(candidate_fn, index=False)
