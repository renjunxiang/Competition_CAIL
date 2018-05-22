from data_transform import data_transform
import json
import pickle
import jieba
import numpy as np

jieba.setLogLevel('WARN')

num_words=20000
maxlen=400
########################################################################################
# train数据集处理
data_transform_train = data_transform()

# 读取json文件
data_train = data_transform_train.read_data(path='./data/data_train.json')

# # 提取需要信息
# data_transform_train.extract_data(name='fact')
# train_fact = data_transform_train.extraction['fact']
#
# #分词并保存原始分词结果，词语长度后期可以再改
# train_fact_cut=data_transform_train.cut_texts(texts=train_fact,word_len=1,need_cut=True,
#                                texts_cut_savepath='./data_deal/data_cut/train_fact_cut.json')
#
# #抽取长度大于1的词语,目的在于去除标点和无意义词
# train_fact_cut_new=data_transform_train.cut_texts(texts=train_fact_cut,word_len=2,need_cut=False,
#                                texts_cut_savepath='./data_deal/data_cut/train_fact_cut_new.json')

with open('./data_deal/data_cut/train_fact_cut_new.json', 'r') as f:
    train_fact_cut_new = json.load(f)

data_transform_train.text2seq(texts_cut=train_fact_cut_new, num_words=num_words, maxlen=maxlen)
tokenizer_fact = data_transform_train.tokenizer_fact
with open('./data_deal/models/tokenizer_fact_%d.pkl'%(num_words),mode='wb') as f:
    pickle.dump(tokenizer_fact,f)
np.save('./data_deal/fact_pad_seq/train_fact_pad_seq_%d_%d.npy'%(num_words,maxlen), data_transform_train.fact_pad_seq)
# train_fact_pad_seq=np.load('./data_cut/train_fact_pad_seq.npy')

# 创建数据one-hot标签
data_transform_train.extract_data(name='accusation')
train_accusations = data_transform_train.extraction['accusation']
data_transform_train.creat_label_set(name='accusation')
train_labels = data_transform_train.creat_labels(name='accusation')
np.save('./data_deal/labels/train_labels_accusation.npy', train_labels)

data_transform_train.extract_data(name='relevant_articles')
train_relevant_articless = data_transform_train.extraction['relevant_articles']
data_transform_train.creat_label_set(name='relevant_articles')
train_labels = data_transform_train.creat_labels(name='relevant_articles')
np.save('./data_deal/labels/train_labels_relevant_articles.npy', train_labels)
########################################################################################
# valid数据集处理
data_transform_valid = data_transform()

# 读取json文件
data_valid = data_transform_valid.read_data(path='./data/data_valid.json')

# # 提取需要信息
# data_transform_valid.extract_data(name='fact')
# valid_fact = data_transform_valid.extraction['fact']
#
# #分词并保存原始分词结果，词语长度后期可以再改
# valid_fact_cut=data_transform_valid.cut_texts(texts=valid_fact,word_len=1,need_cut=True,
#                                texts_cut_savepath='./data_deal/data_cut/valid_fact_cut.json')
#
# #抽取长度大于1的词语,目的在于去除标点和无意义词
# valid_fact_cut_new=data_transform_valid.cut_texts(texts=valid_fact_cut,word_len=2,need_cut=False,
#                                texts_cut_savepath='./data_deal/data_cut/valid_fact_cut_new.json')

with open('./data_deal/data_cut/valid_fact_cut_new.json', 'r') as f:
    valid_fact_cut_new = json.load(f)

data_transform_valid.text2seq(texts_cut=valid_fact_cut_new, tokenizer_fact=tokenizer_fact,
                              num_words=num_words, maxlen=maxlen)
np.save('./data_deal/fact_pad_seq/valid_fact_pad_seq_%d_%d.npy'%(num_words,maxlen), data_transform_valid.fact_pad_seq)

# 创建数据one-hot标签
data_transform_valid.extract_data(name='accusation')
valid_accusations = data_transform_valid.extraction['accusation']
data_transform_valid.creat_label_set(name='accusation')
valid_labels = data_transform_valid.creat_labels(name='accusation')
np.save('./data_deal/labels/valid_labels_accusation.npy', valid_labels)

data_transform_valid.extract_data(name='relevant_articles')
valid_relevant_articless = data_transform_valid.extraction['relevant_articles']
data_transform_valid.creat_label_set(name='relevant_articles')
valid_labels = data_transform_valid.creat_labels(name='relevant_articles')
np.save('./data_deal/labels/valid_labels_relevant_articles.npy', valid_labels)
########################################################################################
# test数据集处理
data_transform_test = data_transform()

# 读取json文件
data_test = data_transform_test.read_data(path='./data/data_test.json')

# # 提取需要信息
# data_transform_test.extract_data(name='fact')
# test_fact = data_transform_test.extraction['fact']
#
# #分词并保存原始分词结果，词语长度后期可以再改
# test_fact_cut=data_transform_test.cut_texts(texts=test_fact,word_len=1,need_cut=True,
#                                texts_cut_savepath='./data_deal/data_cut/test_fact_cut.json')
#
# #抽取长度大于1的词语,目的在于去除标点和无意义词
# test_fact_cut_new=data_transform_test.cut_texts(texts=test_fact_cut,word_len=2,need_cut=False,
#                                texts_cut_savepath='./data_deal/data_cut/test_fact_cut_new.json')

with open('./data_deal/data_cut/test_fact_cut_new.json', 'r') as f:
    test_fact_cut_new = json.load(f)

data_transform_test.text2seq(texts_cut=test_fact_cut_new, tokenizer_fact=tokenizer_fact,
                             num_words=num_words, maxlen=maxlen)
np.save('./data_deal/fact_pad_seq/test_fact_pad_seq_%d_%d.npy'%(num_words,maxlen), data_transform_test.fact_pad_seq)
# test_fact_pad_seq=np.load('./data_cut/test_fact_pad_seq.npy')

# 创建数据one-hot标签
data_transform_test.extract_data(name='accusation')
test_accusations = data_transform_test.extraction['accusation']
data_transform_test.creat_label_set(name='accusation')
test_labels = data_transform_test.creat_labels(name='accusation')
np.save('./data_deal/labels/test_labels_accusation.npy', test_labels)

data_transform_test.extract_data(name='relevant_articles')
test_relevant_articless = data_transform_test.extraction['relevant_articles']
data_transform_test.creat_label_set(name='relevant_articles')
test_labels = data_transform_test.creat_labels(name='relevant_articles')
np.save('./data_deal/labels/test_labels_relevant_articles.npy', test_labels)

# 创建数据标签的集合，用于还原one-hot
np.save('./data_deal/set/set_accusation.npy',data_transform_test.label_set['accusation'])
np.save('./data_deal/set/set_relevant_articles.npy',data_transform_test.label_set['relevant_articles'])