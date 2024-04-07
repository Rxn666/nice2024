import contextlib
import json
import os

from pycocotools.coco import COCO

import pandas as pd
# from lavis.common.dist_utils import main_process
# from lavis.common.registry import registry
# from lavis.tasks.captioning import CaptionTask

from pycocotools.coco import COCO  # isort:skip
from pycocoevalcap.eval import COCOEvalCap  # isort:skip

from vis_utils import view_candidates

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from tqdm import tqdm
import numpy as np

import gc

from utils import merge_res, get_cap_with_best_score, post_processing

def get_candidates(model_name_lis, topk=None, weight_index=0, mask_num=0, candidate_idx_fn=None, mask_fn=None):
    save_dir = f'datasets/nice/eval_res/w{weight_index}'
    save_path = candidate_idx_fn
    if mask_num > -1:
        with open(mask_fn, 'r') as f:
            mask_dict = json.load(f)
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
    return save_path

def score_for_cider_new(split_name):
    nice_gt_root = 'datasets/nice/gt'
    candidate_dir = 'datasets/nice/candidates'
    eval_res_all = []
    dirname = os.path.join('datasets/nice', 'eval_res', model_name)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    for i in tqdm(range(1, 65)):
        candidate_res_file = os.path.join(candidate_dir, f'{i}.json')
        coco_val = caption_eval(nice_gt_root, candidate_res_file, split_name)
        eval_res_col = coco_val.evalImgs
        eval_res_col = [item['CIDEr'] for item in eval_res_col]
        eval_res_all.append(eval_res_col)
    eval_res_all = np.array(eval_res_all)
    eval_res_all_final = eval_res_all.transpose(1, 0)
    col_names = [f'caption{i}' for i in range(1, 65)]
    fn = 'CIDEr'
    save_path = os.path.join('datasets/nice/eval_res', split_name, f'{fn}.csv')
    df = pd.DataFrame(eval_res_all_final, columns=col_names)
    df.to_csv(save_path, index=False)
    
    # with open(os.path.join('nice/output/BLIP2/eval/blip2_opt2.7b_fusion_visual_patch_k_2/20240225004', "score.txt"), "a") as f:
    #     f.write(json.dumps(log_stats) + "\n")

    coco_res = {k: v for k, v in coco_val.eval.items()}
    coco_res["agg_metrics"] = coco_val.eval["CIDEr"]
    return coco_res

def transform_nice_for_cocoeval(ann_path):
    # annotation_file = f"{split_name}.json"
    # ann_path = os.path.join(nice_gt_root, annotation_file)
    with open(ann_path, "r") as f:
        anns = json.load(f)
    assert isinstance(anns, list)

    annotations = []
    images = []
    for i, ann in enumerate(anns):
        image_id = int(ann["image"].split("/")[-1].split(".jpg")[0])
        ann_dict = {"image_id": image_id, "caption": ann["caption"][0], "id": i}
        # ann_dict = {"image_id": image_id, "caption": ann["caption"], "id": i}
        annotations.append(ann_dict)
        images.append({"id": image_id})

    anno_dict = {"images": images, "annotations": annotations}
    out_path = ann_path[:-5] + '_dict.json'
    # with open(os.path.join(nice_gt_root, f"{split_name}_dict.json"), "w") as f:
    with open(out_path, "w") as f:
        json.dump(anno_dict, f)


def caption_eval(results_file, ann_file_coco, sep, scorers):
    if sep:
        # ann_file_coco = os.path.join(nice_gt_root, f"{split}_dict.json")
        ann_file_coco_tran = ann_file_coco[:-5] + '_dict.json'
        if not os.path.isfile(ann_file_coco_tran):
            transform_nice_for_cocoeval(ann_file_coco)
    else:
        ann_file_coco_tran = ann_file_coco

    with contextlib.redirect_stdout(None):
        coco = COCO(ann_file_coco_tran)
        coco_result = coco.loadRes(results_file)
        coco_eval = COCOEvalCap(coco, coco_result) # gt, pred # 改过，之后要改回来
        coco_eval.evaluate(scorers)

    # print output evaluation scores
    print(coco_eval.eval)
    return coco_eval

def to_df(results_file):
    with open(results_file, "r") as f:
        result_json = json.load(f)

    result_dict = {"public_id": [], "caption": []}
    for res in result_json:
        result_dict["public_id"].append(res["image_id"])
        result_dict["caption"].append(res["caption"])

    results = pd.DataFrame.from_dict(result_dict)
    return results

def generate_template():
    tmp = []
    for i in range(7):
        tmp.append([[] for i in range(20000)])
    return  tmp

def score(split_name, sep=False):
    nice_gt_root = 'datasets/nice/gt'
    candidate_dir = 'datasets/nice/candidates'
    # eval_res_all = generate_template()
    dirname = os.path.join('datasets/nice', 'eval_res', model_name)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    metric_lis = ["Bleu", "METEOR", "ROUGE_L", "SPICE"]
    eval_res_all = [[] for i in range(7)]

    for i in tqdm(range(1, 65)):
        candidate_res_file = os.path.join(candidate_dir, f'{i}.json')
        ann_file_coco = f'{nice_gt_root}/{split_name}_dict.json'
        for i in range(len(metric_lis)):
            fn = metric_lis[i]
            print(f'evaluating {fn}')
            scorers = get_scorers(fn)
            coco_val = caption_eval(candidate_res_file, ann_file_coco, sep, scorers)
            eval_res_col = coco_val.evalImgs
            if fn == 'Bleu':
                fn_lis = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
                for i, fn_sub in enumerate(fn_lis):
                    eval_res_col_sub = [item[fn_sub] for item in eval_res_col]
                    eval_res_all[i].append(eval_res_col_sub)
            else:
                if fn == 'SPICE':
                    eval_res_col = [item[fn]['All']['f'] for item in eval_res_col]
                    eval_res_all[-1].append(eval_res_col)
                else:
                    eval_res_col = [item[fn] for item in eval_res_col]
                    if fn == 'METEOR':
                        eval_res_all[4].append(eval_res_col)
                    elif fn == 'ROUGE_L':
                        eval_res_all[5].append(eval_res_col)
                    else:
                        raise NotImplementedError

    col_names = [f'caption{i}' for i in range(1, 65)]
    metric_lis = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "SPICE"]
    for i in range(len(metric_lis)):
        fn = metric_lis[i]
        save_path = os.path.join('datasets/nice/eval_res', split_name, f'{fn}.csv')
        eval_res = eval_res_all[i]
        eval_res = np.array(eval_res)
        eval_res = eval_res.transpose(1, 0)
        df = pd.DataFrame(eval_res, columns=col_names)
        df.to_csv(save_path, index=False)


def score_for_cider_new(split_name):
    nice_gt_root = 'datasets/nice/gt'
    candidate_dir = 'datasets/nice/candidates'
    eval_res_all = []
    dirname = os.path.join('datasets/nice', 'eval_res', model_name)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    ann_path = f'datasets/nice/gt/{split_name}_dict.json'
    fn = 'CIDEr'
    scorer = get_scorers(fn)

    for i in tqdm(range(1, 65)):
        candidate_res_file = os.path.join(candidate_dir, f'{i}.json')
        coco_val = caption_eval(candidate_res_file, ann_path, False, scorer)
        eval_res_col = coco_val.evalImgs
        eval_res_col = [item['CIDEr'] for item in eval_res_col]
        eval_res_all.append(eval_res_col)
    eval_res_all = np.array(eval_res_all)
    eval_res_all_final = eval_res_all.transpose(1, 0)
    col_names = [f'caption{i}' for i in range(1, 65)]
    if not os.path.exists('datasets/nice/eval_res'):
        os.mkdir('datasets/nice/eval_res')
    save_path = os.path.join('datasets/nice/eval_res', split_name, f'{fn}.csv')
    df = pd.DataFrame(eval_res_all_final, columns=col_names)
    df.to_csv(save_path, index=False)
    
    # with open(os.path.join('nice/output/BLIP2/eval/blip2_opt2.7b_fusion_visual_patch_k_2/20240225004', "score.txt"), "a") as f:
    #     f.write(json.dumps(log_stats) + "\n")

    coco_res = {k: v for k, v in coco_val.eval.items()}
    coco_res["agg_metrics"] = coco_val.eval["CIDEr"]
    return coco_res

def convert_sub_to_gt(model_name):
    # in_df = pd.read_csv(f'datasets/nice/model_output/{model_name}.csv')
    in_df = pd.read_excel(f'datasets/nice/model_output/{model_name}.xlsx', engine='openpyxl')
    img_dir = 'test_2024'
    #1. change to gt pre
    gt_pre = []
    for i, row in tqdm(in_df.iterrows()):
        gt_pre_row = {}
        gt_pre_row['caption'] = [row['caption']]
        fn = row['filename']
        gt_pre_row['image'] = f'{img_dir}/{fn}'
        gt_pre.append(gt_pre_row)
    out_path = f'datasets/nice/gt/{model_name}.json'
    with open(out_path, 'w') as f:
        json.dump(gt_pre, f)
    #2. transform to gt coco style
    transform_nice_for_cocoeval(out_path)

def convert_sub_to_gt_new(model_name, img_dir):
    in_df = pd.read_csv(f'datasets/nice/model_output/{model_name}.csv')
    #1. change to gt pre
    gt_pre = []
    for i, row in tqdm(in_df.iterrows()):
        gt_pre_row = {}
        gt_pre_row['caption'] = [row['caption']]
        fn = row['filename']
        gt_pre_row['image'] = f'{img_dir}/{fn}'
        gt_pre.append(gt_pre_row)
    if not os.path.exists('datasets/nice/gt'):
        os.mkdir('datasets/nice/gt')
    with open(f'datasets/nice/gt/{model_name}.json', 'w') as f:
        json.dump(gt_pre, f)

def score_for_cider_between(split_name, split, sep):
    nice_gt_root = 'datasets/nice/gt'
    dirname = os.path.join('datasets/nice', 'eval_res', split_name)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    if split:
        for i in range(2):
            candidate_res_file = os.path.join(nice_gt_root, split_name, f'{i+1}_ref.json')
            ann_file = os.path.join(nice_gt_root, split_name, f'{i+1}.json')
            coco_val = caption_eval(candidate_res_file, ann_file, sep)
            print(f'finish processing {i+1}')
            eval_res_col = coco_val.evalImgs
            eval_res_col = [item['CIDEr'] for item in eval_res_col]
            print('start saving...')
            fn = 'CIDEr'
            with open(os.path.join('datasets/nice/eval_res', split_name, f'{fn}_{i+1}.json'), 'w') as f:
                json.dump(eval_res_col, f)
            gc.collect()
    else:
        candidate_res_file = os.path.join(nice_gt_root, f'{split_name}_ref.json')
        ann_file = os.path.join(nice_gt_root, f'{split_name}.json')
        coco_val = caption_eval(candidate_res_file, ann_file, sep)
        eval_res_col = coco_val.evalImgs
        eval_res_col = [item['CIDEr'] for item in eval_res_col]
        print('start saving...')
        fn = 'CIDEr'
        with open(os.path.join('datasets/nice/eval_res', split_name, f'{fn}.json'), 'w') as f:
            json.dump(eval_res_col, f)

def get_scorers(fn):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]
    if 'Bleu' in fn:
        return [scorers[0]]
    elif fn == 'METEOR':
        return [scorers[1]]
    elif fn == 'ROUGE_L':
        return [scorers[2]]
    elif fn == 'CIDEr':
        return [scorers[3]]
    else:
        return [scorers[4]]


def score_for_intra_similarity(split_name, split, sep, split_num):
    nice_gt_root = 'datasets/nice/gt'
    dirname = os.path.join('datasets/nice', 'eval_res', split_name)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    # metric_lis = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "SPICE"]
    # metric_lis = ["Bleu", "METEOR", "ROUGE_L", "SPICE"]
    metric_lis = ["CIDEr"]
    if split:
        for i in tqdm(range(split_num)):
            print(f'start processing {i+1}')
            candidate_res_file = os.path.join(nice_gt_root, split_name, f'{i+1}_ref.json')
            ann_file = os.path.join(nice_gt_root, split_name, f'{i+1}.json')
            for j in range(len(metric_lis)):
                fn = metric_lis[j]
                print(f'evaluating {fn}')
                scorers = get_scorers(fn)
                coco_val = caption_eval(candidate_res_file, ann_file, sep, scorers)
                eval_res_col = coco_val.evalImgs
                if fn == 'Bleu':
                    fn_lis = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
                    for fn_sub in fn_lis:
                        eval_res_col_sub = [item[fn_sub] for item in eval_res_col]
                        print('saving...')
                        with open(os.path.join('datasets/nice/eval_res', split_name, f'{fn_sub}_{i+1}.json'), 'w') as f:
                            json.dump(eval_res_col_sub, f)
                        gc.collect()
                else:
                    if fn == 'SPICE':
                        eval_res_col = [item[fn]['All']['f'] for item in eval_res_col]
                    else:
                        eval_res_col = [item[fn] for item in eval_res_col]
                    print('saving...')
                    with open(os.path.join('datasets/nice/eval_res', split_name, f'{fn}_{i+1}.json'), 'w') as f:
                        json.dump(eval_res_col, f)
                    gc.collect()
            print(f'finish processing {i+1}')
            
    else:
        candidate_res_file = os.path.join(nice_gt_root, f'{split_name}_ref.json')
        ann_file = os.path.join(nice_gt_root, f'{split_name}.json')
        for i in range(len(metric_lis)):
            fn = metric_lis[i]
            print(f'evaluating {fn}')
            scorers = get_scorers(fn)
            coco_val = caption_eval(candidate_res_file, ann_file, sep, scorers)
            eval_res_col = coco_val.evalImgs
            if fn == 'Bleu':
                fn_lis = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
                for fn_sub in fn_lis:
                    eval_res_col_sub = [item[fn_sub] for item in eval_res_col]
                    print('saving...')
                    with open(os.path.join('datasets/nice/eval_res', split_name, f'{fn_sub}.json'), 'w') as f:
                        json.dump(eval_res_col_sub, f)
                    gc.collect()
            else:
                if fn == 'SPICE':
                    eval_res_col = [item[fn]['All']['f'] for item in eval_res_col]
                else:
                    eval_res_col = [item[fn] for item in eval_res_col]
                print('saving...')
                with open(os.path.join('datasets/nice/eval_res', split_name, f'{fn}.json'), 'w') as f:
                    json.dump(eval_res_col, f)

def post_processing_new(split_name, topk, model_lis, add_model):
    metric_lis = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L"]
    num_model = len(model_lis)
    cols = []
    for model in model_lis:
        for i in range(topk):
            cols.append(model)
    if add_model:
        for model in model_lis:
            cols.append(model)
    if add_model:
        num_cols = num_model*topk + len(model_lis)
    else:
        num_cols = num_model*topk
    for fn in metric_lis:
        with open(os.path.join('datasets/nice/eval_res', split_name, f'{fn}.json'), 'r') as f:
            eval_res = json.load(f)
        eval_res = np.array(eval_res)
        eval_res_all = np.reshape(eval_res, (20000, num_cols))
        save_path = os.path.join('datasets/nice/eval_res', split_name, f'{fn}.csv')
        df = pd.DataFrame(eval_res_all, columns=cols)
        df.to_csv(save_path, index=False)

def get_gt_and_ref(topk):
    # with open('datasets/nice/gt/git_beit3.json', 'r') as f:
    #     tmp = json.load(f)
    # with open('datasets/nice/candidates/1.json', 'r') as f:
    #     ref_tmp = json.load(f)
    can_df = pd.read_csv('datasets/nice/eval_res/w1/vis_id_top5_mask3-10.csv')

    new_gt = []
    new_ref = []
    for i in tqdm(range(20000)):
        # item = tmp[i]
        # item_ref = ref_tmp[i]
        candidate_caps = can_df.loc[i, :].tolist()
        for j in range(2*topk):
            idx = i * (2*topk) + j
            item_ref_new = {}
            item_ref_new['image_id'] = idx
            for k in range(10):
                cap = candidate_caps[k]
                if k == j: 
                    item_ref_new['caption'] = cap
                    continue
                item_new = {}
                item_new["image"] = f'{idx}.jpg'
                item_new['caption'] = [cap]
                new_gt.append(item_new)
            new_ref.append(item_ref_new)
    print(len(new_gt))
    print(len(new_ref))
    with open('datasets/nice/gt/git_beit3_cider_top5.json', 'w') as f:
        json.dump(new_gt, f)
    with open('datasets/nice/gt/git_beit3_cider_top5_ref.json', 'w') as f:
        json.dump(new_ref, f)

def get_gt_and_ref_demo(candidate_caps_fn, model_name): # merge gt
    can_df = pd.read_csv(candidate_caps_fn)
    num_cols = 2

    new_gt = []
    new_ref = []
    image_ids = []
    count = 0
    for i in tqdm(range(20)):
        candidate_caps = can_df.loc[i, :].tolist()
        for j in range(num_cols):
            idx = i * num_cols + j
            item_ref_new = {}
            item_ref_new['image_id'] = idx
            image_ids.append({'id': idx})
            for k in range(num_cols):
                cap = candidate_caps[k]
                if k == j: 
                    item_ref_new['caption'] = cap
                item_new = {}
                item_new["image_id"] = idx
                item_new['caption'] = cap
                item_new['id'] = count
                new_gt.append(item_new)
                count += 1
            new_ref.append(item_ref_new)
    print(len(new_gt))
    print(len(new_ref))
    gt_pack = {}
    gt_pack['annotations'] = new_gt
    gt_pack['images'] = image_ids

    with open(f'datasets/nice/gt/{model_name}.json', 'w') as f:
        json.dump(gt_pack, f)
    with open(f'datasets/nice/gt/{model_name}_ref.json', 'w') as f:
        json.dump(new_ref, f)

def get_gt_and_ref_new(topk, candidate_caps_fn, model_name, model_lis, add_model=False): # merge gt
    model_num = len(model_lis)
    can_df = pd.read_csv(candidate_caps_fn)

    if add_model:
        num_cols = model_num*topk + len(model_lis)
    else:
        num_cols = model_num*topk

    new_gt = []
    new_ref = []
    image_ids = []
    count = 0
    for i in tqdm(range(20000)):
        candidate_caps = can_df.loc[i, :].tolist()
        for j in range(num_cols):
            idx = i * num_cols + j
            item_ref_new = {}
            item_ref_new['image_id'] = idx
            image_ids.append({'id': idx})
            for k in range(num_cols):
                cap = candidate_caps[k]
                if k == j: 
                    item_ref_new['caption'] = cap
                    continue
                item_new = {}
                item_new["image_id"] = idx
                item_new['caption'] = cap
                item_new['id'] = count
                new_gt.append(item_new)
                count += 1
            new_ref.append(item_ref_new)
    print(len(new_gt))
    print(len(new_ref))
    gt_pack = {}
    gt_pack['annotations'] = new_gt
    gt_pack['images'] = image_ids

    with open(f'datasets/nice/gt/{model_name}.json', 'w') as f:
        json.dump(gt_pack, f)
    with open(f'datasets/nice/gt/{model_name}_ref.json', 'w') as f:
        json.dump(new_ref, f)

def get_gt_and_ref_seperately(topk, candidate_caps_fn, model_name, model_lis, add_model):
    model_num = len(model_lis)
    can_df = pd.read_csv(candidate_caps_fn)

    if add_model:
        num_cols = model_num*topk + len(model_lis)
    else:
        num_cols = model_num*topk

    new_gt = []
    new_ref = []
    count = 0
    for i in tqdm(range(20000)):
        candidate_caps = can_df.loc[i, :].tolist()
        for j in range(num_cols):
            for k in range(num_cols):
                item_ref_new = {}
                item_new = {}
                cap = candidate_caps[k]
                ref_cap = candidate_caps[j]
                item_ref_new['caption'] = ref_cap
                if k == j: 
                    continue
                item_new["image"] = f'{count}.jpg'
                item_ref_new['image_id'] = count
                item_new['caption'] = [cap]
                new_gt.append(item_new)
                new_ref.append(item_ref_new)
                count += 1
    print(len(new_gt))
    print(len(new_ref))
    with open(f'datasets/nice/gt/{model_name}.json', 'w') as f:
        json.dump(new_gt, f)
    with open(f'datasets/nice/gt/{model_name}_ref.json', 'w') as f:
        json.dump(new_ref, f)

def get_cap_with_best_cider(model_name, sub_name, candidate_caps_fn):
    in_path = f'datasets/nice/eval_res/w1/{model_name}/CIDEr.csv'
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
    tmp.to_csv(f'datasets/nice/sub/{sub_name}.csv', index=False)

def split_gt_and_ref(model_name, topk, model_lis, add_model, split_count, sep):
    save_dir = f'datasets/nice/gt/{model_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(f'datasets/nice/gt/{model_name}.json', 'r') as f:
        gt = json.load(f)
    with open(f'datasets/nice/gt/{model_name}_ref.json', 'r') as f:
        ref = json.load(f)
    model_num = len(model_lis)
    if add_model:
        num_cols = model_num*topk + len(model_lis)
    else:
        num_cols = model_num*topk
    split_col = num_cols // split_count
    split_num = split_col * (num_cols-1)
    print(split_col)
    print(split_num)
    if sep:
        gt_all = [[] for i in range(split_count)]
        ref_all = [[] for i in range(split_count)]
        count = 0
        for i in tqdm(range(len(gt))):
            count = i % (num_cols * (num_cols - 1))
            gt_sample = gt[i]
            ref_sample = ref[i]
            for j in range(split_count):
                if count >= j * split_num and count < (j+1) * split_num:
                    gt_all[j].append(gt_sample)
                    ref_all[j].append(ref_sample)
        for j in range(split_count):
            print(f'saving split {j+1}')
            with open(f'{save_dir}/{j+1}.json', 'w') as f:
                json.dump(gt_all[j], f)
            with open(f'{save_dir}/{j+1}_ref.json', 'w') as f:
                json.dump(ref_all[j], f)
    else:
        gt1_sub1 = []
        gt1_sub2 = []
        gt3_sub1 = []
        gt3_sub2 = []
        gt2_sub1 = []
        gt2_sub2 = []
        gt4_sub1 = []
        gt4_sub2 = []
        ref_1 = []
        ref_2 = []
        ref_3 = []
        ref_4 = []
        count = 0
        for i in tqdm(range(len(gt['annotations']))):
            count = i % (num_cols * (num_cols - 1))
            # gt_sample = gt[i]
            # ref_sample = ref[i]
            gt_sample = gt['annotations'][i]
            if count < split_num:
                gt1_sub1.append(gt_sample)
                # ref_1.append(ref_sample)
            else:
                if count < 2 * split_num:
                    gt2_sub1.append(gt_sample)
                    # ref_2.append(ref_sample)
                else:
                    if count < 3 * split_num:
                        gt3_sub1.append(gt_sample)
                    else:
                        gt4_sub1.append(gt_sample)
        count = 0
        for i in tqdm(range(len(ref))):
            count = i % num_cols
            ref_sample = ref[i]
            gt_sample_sub = gt['images'][i]
            if count < split_col:
                ref_1.append(ref_sample)
                gt1_sub2.append(gt_sample_sub)
            else:
                if count < 2 * split_col:
                    ref_2.append(ref_sample)
                    gt2_sub2.append(gt_sample_sub)
                else:
                    if count < 3 * split_col:
                        ref_3.append(ref_sample)
                        gt3_sub2.append(gt_sample_sub)
                    else:
                        ref_4.append(ref_sample)
                        gt4_sub2.append(gt_sample_sub)
        gt_1 = {'annotations': gt1_sub1, 'images': gt1_sub2}
        gt_2 = {'annotations': gt2_sub1, 'images': gt2_sub2}
        gt_3 = {'annotations': gt3_sub1, 'images': gt3_sub2}
        gt_4 = {'annotations': gt4_sub1, 'images': gt4_sub2}
        with open(f'{save_dir}/1.json', 'w') as f:
            json.dump(gt_1, f)
        with open(f'{save_dir}/2.json', 'w') as f:
            json.dump(gt_2, f)
        with open(f'{save_dir}/3.json', 'w') as f:
            json.dump(gt_3, f)
        with open(f'{save_dir}/4.json', 'w') as f:
            json.dump(gt_4, f)
        with open(f'{save_dir}/1_ref.json', 'w') as f:
            json.dump(ref_1, f)
        with open(f'{save_dir}/2_ref.json', 'w') as f:
            json.dump(ref_2, f)
        with open(f'{save_dir}/3_ref.json', 'w') as f:
            json.dump(ref_3, f)
        with open(f'{save_dir}/4_ref.json', 'w') as f:
            json.dump(ref_4, f)
        print('finish spliting')

def merge_split_and_post_processing(model_name, topk, model_lis, add_model, split_num, sep):
    model_num = len(model_lis)
    if add_model:
        num_cols = model_num*topk + len(model_lis)
    else:
        num_cols = model_num*topk
    split_col = num_cols // split_num
    cols = []
    fn_lis = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "SPICE", "CIDEr"]
    for model in model_lis:
        for i in range(topk):
            cols.append(model)
    if add_model:
        for model in model_lis:
            cols.append(model)
    for fn in fn_lis:
        data_all = []
        for i in range(split_num):
            with open(os.path.join('datasets/nice/eval_res', model_name, f'{fn}_{i+1}.json'), 'r') as f:
                data = json.load(f)
                data_np = np.array(data)
                if sep:
                    data_np = data_np.reshape((20000, split_col, num_cols-1))
                else:
                    data_np = data_np.reshape((20000, split_col))
                data_all.append(data_np)
        data_all_np = np.concatenate(data_all, axis=1)
        if sep:
            eval_res_avg = np.mean(data_all_np, axis=-1)
        else:
            eval_res_avg = data_all_np
        save_path = os.path.join('datasets/nice/eval_res', model_name, f'{fn}.csv')
        df = pd.DataFrame(eval_res_avg, columns=cols)
        df.to_csv(save_path, index=False)

def get_single_model_similarity(model_lis, img_dir, weights, weight_index):
    for model in model_lis:
        convert_sub_to_gt_new(model, img_dir)
        score_for_cider_new(model)
        merge_res(model_name, weights, weight_index, stage=1)

def intra_similarity_voting(topk, candidate_caps_fn, model_name, model_lis, sub_name, weights, weight_index, mode):
    split_num = 4
    split = True
    add_model = False
    seperate = False
    if mode == 'all':
        # 1. get gt
        get_gt_and_ref_new(topk, candidate_caps_fn, model_name, model_lis)
        # 2 split gt for saving memory
        split_gt_and_ref(model_name, topk, model_lis, add_model, split_num=split_num, sep=False)
        # 3. calculate similarity score
        score_for_intra_similarity(model_name, split, sep=False, split_num=split_num)
        # 4. merge res
        if split:
            merge_split_and_post_processing(model_name, topk, model_lis, add_model, split_num, sep=seperate)
        merge_res(model_name, weights, weight_index, normalized=True, stage=2, model_lis=model_lis)
    else:
        pass
    # 5. output selected cap
    data = get_cap_with_best_score(model_name, sub_name, candidate_caps_fn, weight_index)
    # 6. post processing
    post_processing(data, sub_name)


if __name__ == "__main__":


    "#1. get model similarity res"
    # model_name = 'ofa'
    # 1. conversion    
    # convert_sub_to_gt(model_name)
    # convert_sub_to_gt_new(model_name)

    # 2. eval
    # score(model_name)
    # score_for_cider_new(model_name)

    "#2. select from candidates using cider among them"
    # 最大：20000*20*19
    topk = 5
    mask_num = 3
    split_num = 2
    add_model = False
    split = False
    seperate = True
    model_lis = ['git', 'beit3', 'ofa', 'blip2_large']
    model_name = f'git_beit3_ofa_blip2_cider_top{topk}_mask{mask_num}'
    mask_fn = 'datasets/nice/candidate_captions_unfreq_word_mask_top10.json'
    # sub_name = f'112_w1_top{topk}_beit_4model_mask{mask_num}_cider_among_w1'
    sub_name = 'test'
    candidate_caps_idx_fn = f'datasets/nice/eval_res_final/w1/id_top5_mask{mask_num}-10.csv'
    candidate_caps_fn = f'datasets/nice/eval_res_final/w1/vis_id_top{topk}_mask{mask_num}-10.csv'
    if add_model:
        model_name = model_name + '_add_model'
        sub_name = sub_name + '_add_model'   
        candidate_caps_fn = candidate_caps_fn[:-4] + '_add' + '.csv'
    if seperate:
        model_name = model_name + '_sep'
    if not seperate:
        sub_name = sub_name + '_new'
    
    # 0. prepare candidate_lis
    # get_candidates(model_lis, topk, weight_index=1, mask_num=mask_num, candidate_idx_fn=candidate_caps_idx_fn, mask_fn=mask_fn)
    # view_candidates(candidate_caps_idx_fn, candidate_caps_fn, model_lis, topk, add_model)
    # 1. 
    # get_gt_and_ref_seperately(topk, candidate_caps_fn, model_name, model_lis, add_model)
    # get_gt_and_ref_new(topk, candidate_caps_fn, model_name, model_lis, add_model)
    # 1.5
    # split_gt_and_ref(model_name, topk, model_lis, add_model, split_num, sep=seperate)
    # 2.
    # score_for_cider_between(model_name, split=split, sep=seperate)
    # score_for_others_between(model_name, split=split, sep=seperate, split_num=split_num)
    # 3. 
    if split:
        merge_split_and_post_processing(model_name, topk, model_lis, add_model, split_num, sep=seperate)
    else:
        if seperate:
            post_processing(model_name, topk, model_lis, add_model, split)
        else:
            post_processing_new(model_name, topk, model_lis, add_model)
    # 4. 
    # get_cap_with_best_cider(model_name, sub_name, candidate_caps_fn)
