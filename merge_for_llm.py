import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import math

def split_prompt():
    batch_size = 25
    idx_lis = ['_500_580']
    caps = []
    ignore_batch = []
    # ignore_sample = []
    # count_num = 0
    for idx in idx_lis:
        with open(f'datasets/nice/chatgpt/chatgpt_answer{idx}.json', 'r') as f:
            prompt_lis = json.load(f)
        batch_num = len(prompt_lis)
        tgt_num = batch_size * batch_num
        for i, batch_prompt in enumerate(prompt_lis):
            batch_caps = batch_prompt.split('Task ')
            if len(batch_caps) > batch_size:
                for cap in batch_caps:
                    if cap == '': 
                        continue
                    try: 
                        _, real_cap = cap.split(':')
                        real_cap = real_cap.strip()
                    except: # 处理'1. xxx'
                        try:
                            _, real_cap = cap.split('. ')
                            real_cap = real_cap.strip()
                        except:
                            _, _, real_cap = cap.split(': ')
                            real_cap = real_cap.strip()

                    # item = real_cap.split('.')
                    # if len(item) == 2:
                    #     real_real_cap, _ = item
                    # else:
                    #     real_real_cap1, real_real_cap2, _ = item
                    #     if len(real_real_cap1) > len(real_real_cap2):
                    #         real_real_cap = real_real_cap1
                    #     else:
                    #         real_real_cap = real_real_cap2
                    # real_real_cap = real_real_cap.strip()
                    # real_real_cap = real_real_cap + '.'
                    # assert '\n' not in real_real_cap
                    # caps.append(real_real_cap)
                    try:
                        assert '\n' not in real_cap
                    except:
                        tmp = real_cap.split('\n')
                        real_cap = tmp[0]
                        real_cap = real_cap.strip()
                    if not real_cap.endswith('.'):
                        real_cap = real_cap + '.'
                    caps.append(real_cap)
            else:
                # print(batch_prompt)
                for _ in range(batch_size):
                    caps.append('padding')
                # print(len(batch_caps))
                # print(batch_prompt)
                ignore_batch.append(i)
    print(f'ignore batch: {len(ignore_batch)}')
    num_sample = len(caps)
    assert num_sample == tgt_num
    valid_sample = (batch_num - len(ignore_batch)) * batch_size
    print(f'valid sample: {valid_sample}')
    out_fn = f'datasets/nice/chatgpt_output/chatgpt_answer_split_{num_sample}_{valid_sample}.json'
    with open(out_fn, 'w') as f:
        json.dump(caps, f)
    with open(f'datasets/nice/chatgpt_output/chatgpt_answer_split_{num_sample}_ignore_batch_id.json', 'w') as f:
        json.dump(ignore_batch, f)
    return out_fn, valid_sample

def merge_gpt_res_to_sub(in_fn, baseline_fn, sub_name, start_idx=0):
    with open(in_fn, 'r') as f:
        caps = json.load(f)
    sub_df = pd.read_csv(baseline_fn)
    for i in range(len(caps)):
        cap = caps[i]
        if cap == 'padding': 
            continue
        else:
            sub_df.loc[i+start_idx, 'caption'] = cap
    sub_df.to_csv(sub_name, index=False)
    print('finish gen!')

def cat():
    cols = ['filename']
    for i in range(64):
        cols.append(f'caption{i+1}')
    in_df1 = pd.read_excel('datasets/nice/text_blip_good_bad_0.xlsx', names=cols)
    in_df2 = pd.read_excel('datasets/nice/text_blip_good_bad_1.xlsx', names=cols)
    in_df3 = pd.read_excel('datasets/nice/text_blip_good_bad_2.xlsx', names=cols)
    in_df4 = pd.read_excel('datasets/nice/text_blip_good_bad_3.xlsx', names=cols)

    # merge
    in_df = pd.concat([in_df1, in_df2, in_df3, in_df4])
    in_df.to_csv('datasets/nice/text_blip_good_bad.csv', index=False)

def sort2sub_order():
    fn = 'text_q&a_test'
    in_df = pd.read_csv(f'datasets/nice/{fn}.csv')
    df_filename = in_df['filename'].to_list()

    path_tmp = 'datasets/nice/pred.csv'
    tmp_df = pd.read_csv(path_tmp)
    order = tmp_df['filename'].tolist()

    cols = in_df.columns.values
    out_df = pd.DataFrame(columns=cols)

    for _, filename in tqdm(enumerate(order), total=len(order)):
        i = df_filename.index(filename)
        line = in_df.loc[i]
        out_df.loc[_, :] = line
    
    out_df.to_csv(f'datasets/nice/{fn}_sorted.csv', index=False)

def get_answer_prompt(lis):
    ## not finished
    human_count_col = 'blipvqa-5'
    human_action_col = '?'
    prompt = ''

    for i, ans in lis:
        if i == 0:
            if ans == 'yes':
                prompt = prompt + p1
            else:
                pass
        if i == 1:
            if ans == 'yes':
                prompt = prompt + p2
            else:
                pass
        if i == 2:
            pass
        if i == 3:
            if ans == 'yes':
                prompt = prompt + p4
            else:
                pass

def concat_to_llm_input(topk=3, sample_num=20000, batch_size=0, use_answer=False):
    meta_prompt = f'I have {batch_size} tasks, and the respond answers should in sequence. The respond format should be: Answer 1: ....\n Answer 2: ...\n ....The tasks are as follows: '
    caps_df = pd.read_csv('data/all.csv')
    prompt_start = 'please use merge the following captions into one caption.'
    keywords_df = pd.read_csv('data/text_frequency_dino_sorted_correct.csv')
    prompt_mid = ' output caption should considering the following keywords: '
    prompt_lis = []
    similar_caps_df = pd.read_csv('data/vis_id_top5_mask2-10.csv')
    answer_prompt_df = pd.read_csv('data/vqa_answer.csv')
    # add key info
    answer_prompt_start = 'you should use the following facts. '
    selected_cols = []
    for i in range(2):
        for j in range(3):
            selected_cols.append(i*5 + (5-topk) + j)
    similar_caps = similar_caps_df.iloc[:, np.array(selected_cols)]
    similar_caps = similar_caps.values.tolist()
    count = 1
    ave_len = 0
    prompt_batch = meta_prompt
    num_batch = math.ceil(sample_num/batch_size)
    for i in tqdm(range(sample_num)):
        prompt_batch += f'Task {count}: '
        prompt_sample = prompt_start
        candidate_caps = caps_df.loc[i, :].tolist()
        candidate_caps = candidate_caps + list(set(similar_caps[i]))
        for j in range(len(candidate_caps)):
            prompt_sample += f' caption {j+1}: '
            prompt_sample += candidate_caps[j]
        # 关键词
        # prompt_sample += prompt_mid
        # keywords = keywords_df.loc[i, :].tolist()
        # keywords = keywords[1:]
        # for word in keywords:
        #     if isinstance(word, float): break
        #     prompt_sample += f'{word}, '
        # prompt_sample = prompt_sample[:-2] + '.'
        # 问题的prompt
        if use_answer:
            answer_sample = answer_prompt_df.loc[i, :].tolist()
            answer_prompt = get_answer_prompt(answer_sample)
            prompt_sample = prompt_sample + ' ' + answer_prompt_start
            prompt_sample = prompt_sample + answer_prompt
        prompt_batch += prompt_sample
        prompt_batch += '\n'
        count += 1
        if count > batch_size:
            count = 1
            batch_len = len(prompt_batch.split(' '))
            ave_len += batch_len
            prompt_lis.append(prompt_batch)
            prompt_batch = meta_prompt
        if i == sample_num - 1:
            if len(prompt_lis) == num_batch: pass
            else:
                prompt_lis.append(prompt_batch)
    print('---')
    print(f'average words per batch: {ave_len/num_batch}')
    print('---')
    print(len(prompt_lis))
    with open('prompts.json', 'w') as f:
        json.dump(prompt_lis, f)


if __name__ == "__main__":
    topk = 3 # 两个模型都是top3，一共3+3+2=8个（算上重复）
    sample_num = 20000 # 处理前多少个样本，总共20000
    batch_size = 40
    use_answer = True
    concat_to_llm_input(topk=topk, sample_num=sample_num, batch_size=batch_size, use_answer=use_answer)

    # "post processing..."
    # # 1.
    # out_fn, valid_num = split_prompt()

    # # 2.
    # baseline_fn = 'datasets/nice/sub/55_w1_top5_beit_2model_mask3-10.csv'
    # sub_name = f'datasets/nice/sub/88_w1_top5_beit_2model_mask3-10_chatgpt_{valid_num}_12500+.csv'
    # start_idx = 12500
    # merge_gpt_res_to_sub(out_fn, baseline_fn,sub_name, start_idx=start_idx)