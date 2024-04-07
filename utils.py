import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from vis_utils import view_candidates

weights_all = [
    [0.8, 0.2, 0, 0, 0, 0, 0, 0],
    [0.6, 0.2, 0.15, 0.05, 0, 0, 0, 0],
    [0.5, 0.25, 0.15, 0.1, 0, 0, 0, 0],
    [0.35, 0.25, 0.15, 0.1, 0.08, 0.05, 0.01, 0.01]
]

def get_cider_from_idx():
    in_df = pd.read_csv('datasets/nice/eval_res/w1/id_top5_mask3-10.csv')
    # fn_df = pd.read_csv('datasets/nice/pred.csv')
    # fn_lis = fn_df.loc[:, 'filename']
    cider_git = pd.read_csv('datasets/nice/eval_res/git/CIDEr.csv')
    cider_beit = pd.read_csv('datasets/nice/eval_res/beit3/CIDEr.csv')
    out_cols = in_df.columns.values
    out_df = pd.DataFrame(columns=out_cols, index=range(20000))
    for i in tqdm(range(20000)):
        for j in range(10):
            idx = in_df.iloc[i, j]
            if j < 5:
                cider = cider_git.iloc[i, idx]
            else:
                cider = cider_beit.iloc[i, idx]
            out_df.iloc[i, j] = cider
    out_df.to_csv('datasets/nice/eval_res/w1/vis_id_top5_mask3-10_cider.csv', index=False)

def parse_lis(str):
    str = str[1:-1]
    # str.replace(r', "', )
    lis = str.split("', '")
    lis_new = []
    for item in lis:
        new_item = str.split()
    lis = lis_new
    # new_lis = []
    for i, item in enumerate(lis):
        if i == 0: 
            # lis[i] = item[1:-1]
            lis[i] = item[1:]
        elif i == len(lis)-1:
            lis[i] = item[:-1]
        else:
            # lis[i] = item[2:-1]
            pass
    # for i, item in enumerate(lis):
    #     s_lis = item.split(' ')
    #     new_lis = new_lis + s_lis
    return lis

def parse_dino_word():
    in_df = pd.read_csv('datasets/nice/text_frequency_dino_sorted.csv')
    cols = ['filename']
    for i in range(1, 30):
        cols.append(f'top{i}')
    out_df = pd.DataFrame(columns = cols, index=range(20000))
    for i in tqdm(range(20000)):
        fn = in_df.iloc[i, 0]
        out_df.iloc[i, 0] = fn
        lis = parse_lis(in_df.iloc[i, 1])
        for j, item in enumerate(lis):
            out_df.iloc[i, j+1] = item
    out_df.to_csv('datasets/nice/text_frequency_dino_sorted_correct.csv', index=False)

def rerank():
    img_dir = 'datasets/nice/test_2024'
    ori_order = []
    for fn in os.listdir(img_dir):
        fn = int(fn[:-4])
        ori_order.append(fn)
    ori_order_sorted = sorted(ori_order)

    path_tmp = 'datasets/nice/pred.csv'
    tmp_df = pd.read_csv(path_tmp)
    order = tmp_df['filename'].tolist()
    order = [int(item[:-4]) for item in order]

    in_path = 'datasets/nice/eval_res/w1/beit3.csv'
    out_path = 'datasets/nice/eval_res/w1/beit3_.csv'
    in_df = pd.read_csv(in_path)

    cols = [f'caption{i}' for i in range(1, 65)]
    out_df = pd.DataFrame(columns=cols)
    for _, filename in tqdm(enumerate(order), total=len(order)):
        i = ori_order_sorted.index(filename)
        line = in_df.loc[i]
        out_df.loc[_, :] = line
    out_df.to_csv(out_path)

def change_to_sub_format():

    mode = 'sub' # 'sub' or 'ordered'

    path_tmp = 'datasets/nice/pred.csv'
    tmp_df = pd.read_csv(path_tmp)
    order = tmp_df['filename'].tolist()
    order = [i.replace('.jpg', '') for i in order]
    order = [int(i) for i in order]
    if mode == 'sub':
        pass
    if mode == 'ordered':
        order = sorted(order)

    path = 'datasets/nice/model_output/blip2_large_original.csv'
    # dirname = os.path.dirname(path)
    dirname = 'datasets/nice/model_output'
    df = pd.read_csv(path)

    new_df = pd.DataFrame(columns=['public_id', 'caption'])
    df_filename = df['public_id'].to_list()
    for i, filename in tqdm(enumerate(order), total=len(order)):
        # if i == 10: break
        i = df_filename.index(filename)
        line = df.loc[i]
        new_df = new_df.append(line, ignore_index=True)

    df = new_df
    df['public_id'] = df['public_id'].astype(str).str.replace('$', '.jpg', regex=False)
    df['public_id'] = df['public_id'].apply(lambda x: str(x) + '.jpg')
    df.rename(columns={'public_id': 'filename'}, inplace=True)
    ids = np.arange(190)
    ids2 = np.arange(start=191,stop=20001)
    ids_new = np.concatenate((ids, ids2), axis=0)
    df.insert(loc=0, column='id', value=ids_new)
    if mode == 'sub':
        df.to_csv(os.path.join(dirname, 'blip2_large.csv'), index=False)
    if mode == 'ordered':
        df.to_csv(os.path.join(dirname, 'blip2_large_ordered.csv'), index=False)

def pad_candidate_caps():
    in_df = pd.read_csv('datasets/nice/candidate_captions_sorted_clean.csv')
    out_path = 'datasets/nice/candidate_captions_sorted_clean_pad.csv'

    for i, row in tqdm(in_df.iterrows(), total=20000):
        for j in range(1, 65):
            item = in_df.loc[i, f'caption{j}']
            if isinstance(item, float):
                in_df.loc[i, f'caption{j}'] = 'padding'
    
    in_df.to_csv(out_path, index=False)

def change_candidate_to_coco():
    in_df = pd.read_csv('datasets/nice/candidate_captions_sorted_pad.csv')
    out_json = []
    for i, row in tqdm(in_df.iterrows(), total=20000):
        img_id = row['filename'].split('.')[0]
        for j in range(1, 65):
            item = {}
            cap = row[f'caption{j}']
            item['caption'] = cap
            item['image_id'] = img_id
            out_json.append(item)
    with open('datasets/nice/candidate_captions.json', 'w') as f:
        json.dump(out_json, f)
    return

def split_candidates_coco():
    with open('datasets/nice/candidate_captions.json', 'r') as f:
        in_data = json.load(f)
    splited_data = [[] for _ in range(64)]
    for i in tqdm(range(1280000)):
        index = i % 64
        new_data = in_data[i]
        new_data['image_id'] = int(new_data['image_id'])
        splited_data[index].append(new_data)
    for i in tqdm(range(64)):
        with open(f'datasets/nice/candidates/{i+1}.json', 'w') as f:
            json.dump(splited_data[i], f)

def merge_res(model_name, weights, weight_index=None, normalized=True, stage=None, model_lis=None):
    metrics_lis = ["CIDEr", "SPICE", "METEOR", "ROUGE_L", "Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1"]
    score_lis = []
    save_dir = f'datasets/nice/eval_res/w{weight_index}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cols = []
    if stage == 1:
        cols = [f'caption{i}' for i in range(1, 65)]
    else:
        for model in model_lis:
            for i in range(5):
                cols.append(model)
    for item in tqdm(metrics_lis):
        df = pd.read_csv(os.path.join('datasets/nice/eval_res', model_name, f'{item}.csv'))
        score = df.to_numpy()
        if normalized:
            factor = np.max(score)
            score = score / factor
        score_lis.append(score)
    final_score = 0
    for i in range(8):
        final_score = final_score + weights[i] * score_lis[i]
    final_score_df = pd.DataFrame(final_score, columns=cols)
    final_score_df.to_csv(f'{save_dir}/{model_name}.csv', index=False)

def get_candidates(model_name_lis, topk=None, weight_index=0, mask_num=0, candidate_idx_fn=None, mask_fn=None, candidate_fn=None):
    # pad_mask = np.load('datasets/nice/pad_mask.npy')
    # model_name_lis = ['git', 'beit3', 'ofa', 'blip2_large']
    # topk = 5
    # model_name_lis = ['git_beit3']
    save_dir = f'datasets/nice/eval_res_final/w{weight_index}'
    # save_path = os.path.join(save_dir, candidate_idx_fn)
    save_path = candidate_idx_fn
    with open(mask_fn, 'r') as f:
        mask_dict = json.load(f)
    # with open('datasets/nice/keep_bg_fn_2.json', 'r') as f:
    #     keep_lis = json.load(f)
    id_all = []
    sanity_mask_df = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    for model_name in model_name_lis:
        df = pd.read_csv(f'{save_dir}/{model_name}.csv')
        # df = pd.read_csv(f'datasets/nice/eval_res/{model_name}/SPICE.csv')
        score_np = df.to_numpy()
        # score_sorted = score_np.sort(kind='heapsort')
        for i in tqdm(range(20000)):
            sanity_mask_sample = sanity_mask_df.loc[i, :].tolist()
            sanity_mask_sample = np.array(sanity_mask_sample[1:])
            # pad_mask_sample = pad_mask[i]
            sanity_mask_id = np.where(sanity_mask_sample == 'padding')
            score_np[i, :][sanity_mask_id] = 0
        if mask_num == 'vqa':
            for i in tqdm(range(20000)):
                mask_sample = mask_dict[i]
                tmp = [i for i in range(64)]
                mask_sample_real = list(set(tmp) - set(mask_sample))
                mask_sample_real_np = np.array(mask_sample_real)
                if len(mask_sample_real_np) == 0: continue
                score_np[i, :][mask_sample_real_np] = 0
        elif mask_num > -1:
            mask = mask_dict[str(mask_num)]
            for i in tqdm(range(20000)):
                # mask unfreq
                mask_sample = mask[i]
                mask_sample_np = np.array(mask_sample)
                final_mask_np = mask_sample_np
                # # keep good words
                # keep_mask = keep_lis[i]
                # keep_sample_np = np.array(keep_mask)
                # # diff
                # final_mask_np = np.setdiff1d(mask_sample_np, keep_sample_np)
                if len(final_mask_np) == 0: continue
                score_np[i, :][final_mask_np] = 0
        else:
            pass
        # # keep operation
        # for i in tqdm(range(20000)):
        #     fn = sanity_mask_df.loc[i, 'filename']
        #     if fn in keep_lis:
        #         can_caps = sanity_mask_df.loc[i, :].tolist()[1:]
        #         keep_idx = []
        #         for j, cap in enumerate(can_caps):
        #             if isinstance(cap, float): break
        #             if 'background' in cap:
        #                 keep_idx.append(j)
        #         keep_idx_mask = np.array(keep_idx)
        #         keep_idx_mask= np.isin(range(64), keep_idx_mask)
        #         score_np[i, :] = keep_idx_mask * score_np[i, :]
        #     else:
        #         pass
        indices = score_np.argsort(kind='heapsort')
        selected_id = indices[:, -topk:]
        id_all.append(selected_id)
    id_all = np.concatenate(id_all, 1)
    # id_all = id_all.transpose(1, 0)
    cols = []
    for i in model_name_lis:
        for j in range(topk):
            cols.append(i)
    out_df = pd.DataFrame(id_all, columns=cols)
    out_df.to_csv(save_path, index=False)
    view_candidates(candidate_idx_fn, candidate_fn, model_name_lis, topk, add_model=False)
    return save_path

def sort2sub_order(): # badd!!!!!!!!
    in_df = pd.read_csv('datasets/nice/model_output/beit3.csv')
    df_filename = in_df['filename'].to_list()

    path_tmp = 'datasets/nice/pred.csv'
    tmp_df = pd.read_csv(path_tmp)
    order = tmp_df['filename'].tolist()

    cols = ['id', 'filename', 'caption']
    out_df = pd.DataFrame(columns=cols)

    for _, filename in tqdm(enumerate(order), total=len(order)):
        i = df_filename.index(filename)
        line = in_df.loc[i]
        if _ <190:
            line['id'] = _
        else:
            line['id'] = _ + 1
        out_df.loc[_, :] = line
    
    out_df.to_csv('datasets/nice/model_output/beit3_.csv', index=False)

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

def get_best_id(sample_id, topk):
    score_dict = {}
    for i, idx in enumerate(sample_id):
        score = i % topk + 1
        if idx not in score_dict:
            score_dict[idx] = score
        else:
            score_dict[idx] = score_dict[idx] + score
    sorted_dict = sorted(score_dict.items(), key=lambda x:x[1])
    return sorted_dict[-1][0]

def voting_and_formatting(topk, best_model, sub_name=None, model_name_lis=None, candidate_idx_fn=None):
    if topk <= 5:
        topk_tmp = 5
    best_col = best_model * topk - 1 
    df = pd.read_csv(candidate_idx_fn)
    id_np = df.to_numpy()
    used_col = model_to_col(model_name_lis, topk, topk_tmp)
    print(used_col)
    id_np = id_np[:, used_col]
    count = 0
    selected_id = []
    for i, sample_id in tqdm(enumerate(id_np)):
        unique_id, count_id = np.unique(sample_id, return_counts=True)
        if len(unique_id) < len(sample_id):
            count += 1
            best_id = get_best_id(sample_id, topk)
            selected_id.append(best_id)
            # best_id = np.argmax(count_id)
            # selected_id.append(unique_id[best_id])
        else:
            selected_id.append(sample_id[best_col])
    tmp = pd.read_csv('datasets/nice/pred.csv')
    candidate_caps = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    candidate_caps = candidate_caps.values.tolist()
    selected_caps = []
    for i in range(20000):
        cap = candidate_caps[i][selected_id[i]+1]
        selected_caps.append(cap)
    tmp['caption'] = selected_caps
    tmp.to_csv(f'datasets/nice/sub/{sub_name}.csv', index=False)
    print(f'count voting: {count}')

def merge_model_output(model_name_lis):
    save_dir = 'datasets/nice/model_output'
    out_df = pd.DataFrame()
    for model_name in model_name_lis:
        df = pd.read_csv(f'{save_dir}/{model_name}.csv')
        caps = df['caption'].tolist()
        out_df[model_name] = caps
    out_df.to_csv(f'{save_dir}/all.csv', index=False)

def select_single_col(sub_name, topk=0, weight_index=0, best_col=None, mask_num=None):
    in_path = f'datasets/nice/eval_res/w{weight_index}/id_top{topk}_cider2_mask{mask_num}.csv'
    score_id = pd.read_csv(in_path)
    score_id_np = score_id.to_numpy()
    selected_ids = score_id_np.transpose(1, 0)
    selected_ids = selected_ids[best_col, :]
    tmp = pd.read_csv('datasets/nice/pred.csv')
    candidate_caps = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    candidate_caps = candidate_caps.values.tolist()
    selected_caps = []
    for i in tqdm(range(20000)):
        cap = candidate_caps[i][selected_ids[i]+1]
        selected_caps.append(cap)
    tmp['caption'] = selected_caps
    tmp.to_csv(f'datasets/nice/sub/{sub_name}.csv', index=False)

def check_ranking_exclude(sub_name):
    candidate_df = pd.read_csv('datasets/nice/candidate_captions_sorted_clean.csv')
    target_df = pd.read_csv(f'datasets/nice/sub/{sub_name}.csv')
    for i in tqdm(range(20000)):
        target = target_df.loc[i, 'caption']
        candidates = candidate_df.loc[i, :].to_list()
        if target in candidates:
            pass
        else:
            print(f'fail at line {i}')

def score_cap_with_freq_words():
    text_freq_df = pd.read_csv('datasets/nice/text_frequency_sorted.csv')
    in_df = pd.read_csv('datasets/nice/candidate_captions_sorted_utf8.csv')
    out_df = pd.read_csv('datasets/nice/candidate_captions_sorted_utf8.csv')
    for i in tqdm(range(20000)):
        text_freq = text_freq_df.loc[i, :].tolist()
        text_freq = text_freq[1:]
        can_caps = in_df.loc[i, :].tolist()
        can_caps = can_caps[1:]
        for j, cap in enumerate(can_caps):
            if isinstance(cap, float): continue
            score = 0
            for k, word in enumerate(text_freq):
                if word in cap:
                    score += 5-k
            out_df.iloc[i, j+1] = score
    out_df.to_csv('datasets/nice/candidate_captions_sorted_text_freq_score.csv', index=False)

def get_cap_with_freq_word_score():
    in_df = pd.read_csv('datasets/nice/candidate_captions_sorted_text_freq_score.csv')
    can_df = pd.read_csv('datasets/nice/candidate_captions_sorted_utf8.csv')
    in_df = in_df.iloc[:, 1:]
    in_df = in_df.to_numpy()
    m_len = 0
    res = []
    for score in in_df:
        max_score = np.nanmax(score)
        max_id = np.where(score == max_score)
        len_id = len(max_id[0])
        if len_id > m_len:
            m_len = len_id
        res.append(max_id[0].tolist())
    print(f'max_len:{m_len}')
    with open('datasets/nice/candidate_captions_sorted_with_best_text_freq_score.json', 'w') as f:
        json.dump(res, f)

def get_cap_with_unfreq_word_dict():
    text_freq_df1 = pd.read_csv('datasets/nice/text_frequency_top10_sorted.csv')
    text_freq_df2 = pd.read_csv('datasets/nice/model_output/git_beit3_used_words.csv')
    in_df = pd.read_csv('datasets/nice/candidate_captions_sorted_utf8.csv')
    mask_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    for i in tqdm(range(20000)):
        text_freq1 = text_freq_df1.loc[i, :].tolist()
        text_freq1 = text_freq1[1:]
        text_freq2 = text_freq_df2.loc[i, :].tolist()
        text_freq = text_freq1 + text_freq2
        text_freq = list(set(text_freq))
        can_caps = in_df.loc[i, :].tolist()
        can_caps = can_caps[1:]
        tmp_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        for j, cap in enumerate(can_caps):
            if isinstance(cap, float): continue
            count = 0
            for k, word in enumerate(text_freq):
                if isinstance(word, float): continue
                if word in cap:
                    count += 1
            if count == 0:
                tmp_dict[0].append(j)
                tmp_dict[1].append(j)
                tmp_dict[2].append(j)
                tmp_dict[3].append(j)
                tmp_dict[4].append(j)
                tmp_dict[5].append(j)
            elif count == 1:
                tmp_dict[1].append(j)
                tmp_dict[2].append(j)
                tmp_dict[3].append(j)
                tmp_dict[4].append(j)
                tmp_dict[5].append(j)
            elif count == 2:
                tmp_dict[2].append(j)
                tmp_dict[3].append(j)
                tmp_dict[4].append(j)
                tmp_dict[5].append(j)
            elif count == 3:
                tmp_dict[3].append(j)
                tmp_dict[4].append(j)
                tmp_dict[5].append(j)
            elif count == 4:
                tmp_dict[4].append(j)
                tmp_dict[5].append(j)
            elif count == 5:
                tmp_dict[5].append(j)
        mask_dict[0].append(tmp_dict[0])
        mask_dict[1].append(tmp_dict[1])
        mask_dict[2].append(tmp_dict[2])
        mask_dict[3].append(tmp_dict[3])
        mask_dict[4].append(tmp_dict[4])
        mask_dict[5].append(tmp_dict[5])
    with open('datasets/nice/candidate_captions_unfreq_word_mask_top10_git_beit.json', 'w') as f:
        json.dump(mask_dict, f)

def sanity_check():
    in_df1 = pd.read_csv('datasets/nice/candidate_captions_sorted.csv')
    in_df2 = pd.read_csv('datasets/nice/candidate_captions_sorted_utf8.csv')
    out_df = pd.read_csv('datasets/nice/candidate_captions_sorted_utf8.csv')
    for i in tqdm(range(20000)):
        for j in range(65):
            data1 = in_df1.iloc[i, j]
            data2 = in_df2.iloc[i, j]
            if data1 == data2:
                pass
            else:
                if isinstance(data1, float): continue
                print(data1)
                print(data2)
                out_df.iloc[i, j] = 'padding'
    out_df.to_csv('datasets/nice/candidate_captions_sorted_clean.csv', index=False)

def get_padding_mask():
    in_df = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    data = in_df.to_numpy()
    data = data[:, 1:]
    mask = data == 'padding'
    mask_reverse = ~ mask
    np.save('datasets/nice/pad_mask.npy', mask_reverse)

def get_keep_word_mask():
    keep_word_lis = ['background', 'model', 'symbol', 'abstract', 'recycle']
    in_df = pd.read_csv('datasets/nice/candidate_captions_sorted_utf8.csv')
    unmask_lis = []
    for i in tqdm(range(20000)):
        can_caps = in_df.loc[i, :].tolist() 
        can_caps = can_caps[1:]
        unmask_lis_line = []
        for j, cap in enumerate(can_caps):
            if isinstance(cap, float): continue
            for word in keep_word_lis:
                if word in cap:
                    unmask_lis_line.append(j)
                    break
        unmask_lis.append(unmask_lis_line)
    with open('datasets/nice/keep_word_mask_2.json', 'w') as f:
        json.dump(unmask_lis, f)

def remove_dino_bad_words():
    people_asso_words = ['man', 'men', 'woman', 'women', 'girl', 'male', 'female', \
        'daughter', 'boy', 'son', 'people', 'children', 'kids', 'boys', 'girls', 'father', 'mother']
    in_df = pd.read_csv('datasets/nice/text_frequency_dino_sorted_correct.csv')
    cols = in_df.columns.values
    out_df = pd.DataFrame(columns=cols, index=range(20000))
    for i in tqdm(range(20000)):
        row = in_df.loc[i, :].tolist()
        out_df.iloc[i, 0] = row[0]
        keywords = row[1:]
        count = 0
        for j, keyword in enumerate(keywords):
            if isinstance(keyword, float): break
            if keyword not in people_asso_words:
                out_df.iloc[i, count+1] = keyword
                count += 1
    out_df.to_csv('datasets/nice/text_frequency_dino_sorted_correct_filtered.csv', index=False)

# get caps of best cider (ave 2 models)
def prepare_for_cider_two_ref():
    with open('datasets/nice/gt/beit3.json', 'r') as f:
        data1 = json.load(f)
    with open('datasets/nice/gt/git.json', 'r') as f:
        data2 = json.load(f)
    for i in range(20000):
        new_cap = data2[i]['caption'][0]
        data1[i]['caption'][0] = data1[i]['caption'][0] + '\n' + new_cap
    with open('datasets/nice/gt/git_beit3.json', 'w') as f:
        json.dump(data1, f)

def get_cap_for_average_2_model_cider():
    save_path = 'datasets/nice/eval_res/git_beit3/CIDEr_sep.csv'
    in_df1 = pd.read_csv('datasets/nice/eval_res/git/CIDEr.csv')
    in_df2 = pd.read_csv('datasets/nice/eval_res/beit3/CIDEr.csv')
    cols = in_df1.columns.values
    out_df = pd.DataFrame(columns=cols, index=range(20000))
    for i in tqdm(range(20000)):
        for j in range(64):
            cider1 = in_df1.iloc[i, j]
            cider2 = in_df2.iloc[i, j]
            avg_cider = (cider1 + cider2) / 2
            out_df.iloc[i, j] = avg_cider
    out_df.to_csv(save_path, index=False)

def select_using_avg_cider(model_name=None, mask_num=0, mask_fn=None, sub_name=None):
    if mask_num > -1:
        with open(mask_fn, 'r') as f:
            mask_dict = json.load(f)
    sanity_mask_df = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    df = pd.read_csv(f'datasets/nice/eval_res/{model_name}/CIDEr_sep.csv')
    score_np = df.to_numpy()
    for i in tqdm(range(20000)):
        sanity_mask_sample = sanity_mask_df.loc[i, :].tolist()
        sanity_mask_sample = np.array(sanity_mask_sample[1:])
        sanity_mask_id = np.where(sanity_mask_sample == 'padding')
        score_np[i, :][sanity_mask_id] = 0
    if mask_num > -1:
        mask = mask_dict[str(mask_num)]
        for i in tqdm(range(20000)):
            # mask unfreq
            mask_sample = mask[i]
            mask_sample_np = np.array(mask_sample)
            final_mask_np = mask_sample_np
            if len(final_mask_np) == 0: continue
            score_np[i, :][final_mask_np] = 0
    else:
        pass
    indices = score_np.argsort(kind='heapsort')
    selected_ids = indices[:, -1]
    tmp = pd.read_csv('datasets/nice/pred.csv')
    candidate_caps = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    candidate_caps = candidate_caps.values.tolist()
    selected_caps = []
    for i in tqdm(range(20000)):
        cap = candidate_caps[i][selected_ids[i]+1]
        selected_caps.append(cap)
    tmp['caption'] = selected_caps
    tmp.to_csv(f'datasets/nice/sub/{sub_name}.csv', index=False)

# model_name = 'git_beit3'
# mask_num = 3
# mask_fn = f'datasets/nice/candidate_captions_unfreq_word_mask_top10.json'
# sub_name = f'82_git_beit3_cider_avg_mask_{mask_num}-10'
# select_using_avg_cider(model_name, mask_num, mask_fn, sub_name)

# get model with vqa mask
def get_blip_vqa_mask():
    # in_df1 = pd.read_csv('datasets/nice/text_blip_good_bad_1.csv')
    # in_df2 = pd.read_csv('datasets/nice/text_blip_good_bad_2.csv')

    # # merge
    # in_df = pd.concat([in_df1, in_df2])
    # in_df.to_csv('datasets/nice/text_blip_good_bad_1.csv', index=False)
    in_df = pd.read_csv('datasets/nice/text_blip_good_bad.csv')
    df_filename = in_df['filename'].tolist()

    path_tmp = 'datasets/nice/pred.csv'
    tmp_df = pd.read_csv(path_tmp)
    order = tmp_df['filename'].tolist()

    candidate_caps_df = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')

    selected_idx_all = []

    for _, filename in tqdm(enumerate(order), total=len(order)):
        i = df_filename.index(filename)
        line = in_df.loc[i, :].values.tolist()
        selected_caps = line[1:]
        candidate_caps = candidate_caps_df.loc[_, :].tolist()
        candidate_caps = candidate_caps[1:]
        selected_idx = []
        for cap in selected_caps:
            if isinstance(cap, float): continue
            if cap == 'padding': continue
            try:
                idx = candidate_caps.index(cap)
                selected_idx.append(idx)
            except:
                pass
        selected_idx_all.append(selected_idx)
        
    
    with open('datasets/nice/candidate_captions_blip_vqa_mask.json', 'w') as f:
        json.dump(selected_idx_all, f)

# get model similar cap with best cider
def select_model_best(sub_name, candidate_idx_fn):
    in_path = candidate_idx_fn
    top_idx = pd.read_csv(in_path)
    cider_df = pd.read_csv('datasets/nice/eval_res/w1/vis_id_top5_mask3-10_cider.csv')

    tmp = pd.read_csv('datasets/nice/pred.csv')
    candidate_caps = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    candidate_caps = candidate_caps.values.tolist()

    selected_caps = []
    for i in tqdm(range(20000)):
        cider_git = cider_df.iloc[i, 4]
        cider_beit = cider_df.iloc[i, 9]
        if cider_git > cider_beit:
            best_col = 4
        else:
            best_col = 9
        selected_id = top_idx.iloc[i, best_col]
        cap = candidate_caps[i][selected_id+1]
        selected_caps.append(cap)
    tmp['caption'] = selected_caps
    tmp.to_csv(f'datasets/nice/sub/{sub_name}.csv', index=False)

# get model most similar cap before fusing
def get_single_model_best(): # only CIDEr!!!!
    model_name = 'mplug'
    sub_name = f'102_single_{model_name}_ranking_cider'
    in_path =  f'datasets/nice/eval_res/{model_name}/CIDEr.csv'
    sanity_mask_df = pd.read_csv('datasets/nice/candidate_captions_sorted_clean_pad.csv')
    df = pd.read_csv(in_path)
    score_np = df.to_numpy()
    # score_sorted = score_np.sort(kind='heapsort')
    for i in tqdm(range(20000)):
        sanity_mask_sample = sanity_mask_df.loc[i, :].tolist()
        sanity_mask_sample = np.array(sanity_mask_sample[1:])
        sanity_mask_id = np.where(sanity_mask_sample == 'padding')
        score_np[i, :][sanity_mask_id] = 0
    indices = score_np.argsort(kind='heapsort')
    selected_id = indices[:, -1]

    tmp = pd.read_csv('datasets/nice/pred.csv')
    selected_caps = []
    for i in range(20000):
        cap = sanity_mask_df.iloc[i, selected_id[i]+1]
        selected_caps.append(cap)
    tmp['caption'] = selected_caps
    tmp.to_csv(f'datasets/nice/sub/{sub_name}.csv', index=False)

def get_cap_with_best_score(model_name, sub_name, candidate_caps_fn, weight_index):
    in_path = f'datasets/nice/eval_res/w{weight_index}/{model_name}.csv'
    df = pd.read_csv(in_path)
    candidate_df = pd.read_csv(candidate_caps_fn)
    score_np = df.to_numpy()
    indices = score_np.argsort(kind='heapsort')
    selected_ids = indices[:, -1]
    tmp = pd.read_csv('datasets/nice/pred.csv')
    selected_caps = []
    for i in tqdm(range(20000)):
        cap = candidate_df.iloc[i, selected_ids[i]]
        selected_caps.append(cap)
    tmp['caption'] = selected_caps
    return tmp
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    tmp.to_csv(f'{output_dir}/{sub_name}.csv', index=False)

def post_processing(data, sub_name):
    ref_path = 'datasets/nice/beit3_retrieval.csv'
    if not os.path.exists('output'):
        os.mkdir('output')
    out_path = f'output/{sub_name}.csv'
    in_df = data
    ref_df = pd.read_csv(ref_path)
    for i in range(20000):
        cap = in_df.loc[i, 'caption']
        if cap == 'padding':
            new_cap = ref_df.loc[i, 'caption']
            in_df.loc[i, 'caption'] = new_cap
    in_df.to_csv(out_path, index=False)

def main_process():
    model_name_lis = ['git', 'beit3']
    # model_name_lis = ['git', 'beit3']
    weight_index = 1
    weights = weights_all[weight_index-1]
    topk = 5
    best_model = 2
    mask_num = 3

    sub_name = f'112_w{weight_index}_top{topk}_beit_2model_mask{mask_num}-10_keep_bg'
    candidate_idx_fn = f'datasets/nice/eval_res_final/w{weight_index}/id_top5_mask{mask_num}-10.csv'
    candidate_fn = f'datasets/nice/eval_res_final/w{weight_index}/vis_id_top5_mask{mask_num}-10.csv'
    mask_fn = f'datasets/nice/candidate_captions_unfreq_word_mask_top10.json'

    # 1. get_final_score_per_model
    # for model_name in model_name_lis:
    #     merge_res(model_name, weights, weight_index, stage=1)

    # 1.5 get_bad_caption_mask
    # get_cap_with_unfreq_word_dict()

    # 2. get_best_caps_per_model
    # candidate_idx_fn = get_candidates(model_name_lis, topk=topk, weight_index=weight_index, mask_num=mask_num, candidate_idx_fn=candidate_idx_fn, mask_fn=mask_fn)
    # view_candidates(candidate_idx_fn, candidate_fn, model_name_lis, topk, False)

    # 3. voting
    voting_and_formatting(topk=topk, best_model=best_model, sub_name=sub_name, model_name_lis=model_name_lis, candidate_idx_fn=candidate_idx_fn)
    # select_single_col(sub_name, topk, weight_index, best_col=4, mask_num=mask_num)
    # select_model_best(sub_name=sub_name, candidate_idx_fn=candidate_idx_fn)

    # 4. checking
    check_ranking_exclude(sub_name)

def main_score_among():
    weight_index = 3
    weights = weights_all[weight_index-1]
    model_name = 'git_beit3_ofa_blip2_cider_top5_mask3_sep'
    sub_name = f'111_w{weight_index}_top5_beit_4model_mask3_score_among'
    candidate_caps_fn = f'datasets/nice/eval_res/w1/vis_id_top5_mask3-10_git_beit3_ofa_blip2.csv'
    #1. 
    merge_res(model_name, weights, weight_index, normalized=True, stage=2)
    #2.
    get_cap_with_best_score(model_name, sub_name, candidate_caps_fn, weight_index)

# main_process()
# main_score_among()