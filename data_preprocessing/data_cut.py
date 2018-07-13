from data_transform import data_transform
import json
import pickle
import jieba
import numpy as np

jieba.setLogLevel('WARN')

num_words = 40000
maxlen = 400
########################################################################################
# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')

# 提取需要信息
data_transform_big.extract_data(name='fact')
# big_fact = data_transform_big.extraction['fact']

# 分词并保存原始分词结果，词语长度后期可以再改
for i in range(18):
    texts=data_transform_big.extraction['fact'][i*100000:(i*100000 + 100000)]
    big_fact_cut = data_transform_big.cut_texts(texts=texts, word_len=1,
                                                need_cut=True)
    with open('./data_deal/data_cut/big_fact_cut_%d_%d.pkl' % (i*100000, i*100000 + 100000), mode='wb') as f:
        pickle.dump(big_fact_cut, f)
    print('finish big_fact_cut_%d_%d' % (i*100000, i*100000 + 100000))

for i in range(18):
    print('start big_fact_cut_%d_%d' % (i*100000, i*100000 + 100000))
    with open('./data_deal/data_cut/big_fact_cut_%d_%d.pkl' % (i*100000, i*100000 + 100000), mode='rb') as f:
        big_fact_cut = pickle.load(f)
    data_transform_big = data_transform()
    big_fact_cut_new = data_transform_big.cut_texts(texts=big_fact_cut,
                                                    word_len=2,
                                                    need_cut=False)
    with open('./data_deal/data_cut/big_fact_cut_%d_%d_new.pkl' % (i*100000, i*100000 + 100000), mode='wb') as f:
        pickle.dump(big_fact_cut_new, f)
    print('finish big_fact_cut_%d_%d' % (i*100000, i*100000 + 100000))
