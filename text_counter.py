from tqdm import tqdm
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
import nltk



def split_words():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # 读取CSV文件
    # data = pd.read_csv('/home/yqx/yqx_softlink/image2caption/candidate_captions_pad.csv')
    data = pd.read_csv('datasets/nice/model_output/all.csv')
    # 助动词列表，你可以根据实际需要扩展
    auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had',
                    'do', 'does', 'did', 'shall', 'will', 'should', 'would', 'may', 'might', 'must',
                    'can', 'could']
    # 初始化一个空的DataFrame来存储结果
    # result_df = pd.DataFrame(columns=['filename', 
    #                                   ])
    model_lis = ['git', 'beit3']
    for model in model_lis:
        out_path = f'datasets/nice/model_output/{model}_split.csv'
        cols = []
        for i in range(30):
            cols.append(f'word{i+1}')
        out_df = pd.DataFrame(columns=cols, index=range(20000))

        # 遍历每一行数据
        for index, row in tqdm(data.iterrows(), total=20000):
            # 将所有描述合并成一个文本
            descriptions = row[model]
            # combined_text = ' '.join(descriptions)
            combined_text = descriptions

            # 分词并进行词性标注
            tokens = word_tokenize(combined_text)
            tagged_tokens = pos_tag(tokens)

            # 只统计名词、动词和形容词的词频，并排除助动词
            valid_words = [word for word, pos in tagged_tokens if (
                    pos.startswith('NN') or pos.startswith('VB') or pos.startswith(
                'JJ')) and word.lower() not in auxiliary_verbs]
            
            for i, word in enumerate(valid_words):
                out_df.iloc[index, i] = word


            # 统计词频
            # fdist = FreqDist(valid_words)

            # 获取词频最高的五个词汇
            # top_words = fdist.most_common(50)
        out_df.to_csv(out_path, index=False)

    # 将结果写入新的CSV文件
    # result_df.to_csv('text_frequency_top50.csv', index=False)

def merge_split_words():
    in_df1 = pd.read_csv('datasets/nice/model_output/beit3_split.csv')
    in_df2 = pd.read_csv('datasets/nice/model_output/git_split.csv')
    cols = [f'word{i}' for i in range(1, 31)]
    out_df = pd.DataFrame(columns=cols, index=range(20000))
    for i in range(20000):
        word_lis1 = in_df1.loc[i, :]
        word_lis2 = in_df2.loc[i, :]
        word_merge = set(word_lis1) | set(word_lis2)
        count = 0
        for j, item in enumerate(list(word_merge)):
            if isinstance(item, float): continue
            out_df.iloc[i, count] = item
            count += 1
    out_df.to_csv('datasets/nice/model_output/git_beit3_used_words.csv', index=False)

merge_split_words()