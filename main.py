from similarity_utils import get_single_model_similarity, intra_similarity_voting
from utils import get_candidates
import argparse

WEIGHTS = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0.8, 0.2, 0, 0, 0, 0, 0, 0],
    [0.6, 0.2, 0.15, 0.05, 0, 0, 0, 0],
    [0.5, 0.25, 0.15, 0.1, 0, 0, 0, 0],
    [0.35, 0.25, 0.15, 0.1, 0.08, 0.05, 0.01, 0.01]
]

## setup: set hyper parameters
model_lis = ['git', 'beit3', 'ofa', 'blip2_large']
img_dir = 'test_2024'
topk = 5
weight_index = 1
weights = WEIGHTS[weight_index-1]
mask_num = 3
candidate_idx_fn = f'datasets/nice/eval_res/w{weight_index}/id_top{topk}_mask{mask_num}-10.csv'
candidate_fn = f'checkpoints/w{weight_index}/vis_id_top{topk}_mask{mask_num}-10.csv'
mask_fn = 'datasets/nice/candidate_captions_unfreq_word_mask_top10.json'
model_name = f'git_beit3_ofa_blip2_top{topk}_mask{mask_num}'
sub_name = 'output'


parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--mode', type=str, default='quick', help='whether go through all procedure.')
args = parser.parse_args()

# if args.mode == 'all':
#     ## 0. preparing (optional)

#     # inference pretrained models
#     # get high frequency words
#     # get vqa and grounding infomation

#     # 1. filtering and retrieval

#     # scoring for candidates based on model output
#     get_single_model_similarity(model_lis, img_dir, weights, weight_index) # 问题：函数名没改，没传weight

#     # retrieve topk candidates and filtering
#     print('performing retrieval...')
#     get_candidates(model_lis, topk, weight_index, mask_num, candidate_idx_fn, mask_fn, candidate_fn) # 问题：需要增加keep_file

# else:
#     pass

## 2. voting
intra_similarity_voting(topk, candidate_fn, model_name, model_lis, sub_name, weights, weight_index, mode = args.mode)




