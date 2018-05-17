from data_transform import data_transform
import json
import time
import jieba
jieba.setLogLevel('WARN')
import numpy as np
########################################################################################
# train数据集处理
data_transform_train=data_transform()

# 读取json文件
data_train = data_transform_train.read_data(path='./data/data_train.json')

# 提取需要信息
data_transform_train.extract_data(name='fact')
train_fact = data_transform_train.extraction['fact']

# #提取需要信息(分词速度太慢，保存分词结果)
# train_fact_cut=[jieba.lcut(i) for i in train_fact]
# with open('./data_cut/train_fact_cut.json','w') as f:
#     json.dump(train_fact_cut,f)

with open('./data_cut/train_fact_cut.json','r') as f:
    train_fact_cut=json.load(f)

train_fact_pad_seq=data_transform_train.text2seq(texts=train_fact_cut,needcut=False,num_words=10000, maxlen=400)
tokenizer_fact = data_transform_train.tokenizer_fact
np.save('./data_cut/train_fact_pad_seq.npy',train_fact_pad_seq)
# train_fact_pad_seq=np.load('./data_cut/train_fact_pad_seq.npy')
data_transform_train.extract_data(name='accusation')
train_accusations = data_transform_train.extraction['accusation']
data_transform_train.creat_label_set(name='accusation')
data_transform_train.creat_labels(name='accusation')
train_labels=np.array(data_transform_train.labels_one_hot)
np.save('./data_cut/train_labels.npy',train_labels)
########################################################################################
# valid数据集处理
data_transform_valid=data_transform()

# 读取json文件
data_valid = data_transform_valid.read_data(path='./data/data_valid.json')

# 提取需要信息
data_transform_valid.extract_data(name='fact')
valid_fact = data_transform_valid.extraction['fact']

# #提取需要信息(分词速度太慢，保存分词结果)
# valid_fact_cut=[jieba.lcut(i) for i in valid_fact]

with open('./data_cut/valid_fact_cut.json','r') as f:
    valid_fact_cut=json.load(f)

valid_fact_pad_seq=data_transform_valid.text2seq(texts=valid_fact_cut,needcut=False,
                                                 tokenizer_fact=tokenizer_fact,num_words=10000, maxlen=400)
np.save('./data_cut/valid_fact_pad_seq.npy',valid_fact_pad_seq)
# valid_fact_pad_seq=np.load('./data_cut/valid_fact_pad_seq.npy')
data_transform_valid.extract_data(name='accusation')
valid_accusations = data_transform_valid.extraction['accusation']
data_transform_valid.creat_label_set(name='accusation')
data_transform_valid.creat_labels(name='accusation')
valid_labels=np.array(data_transform_valid.labels_one_hot)
np.save('./data_cut/valid_labels.npy',valid_labels)
########################################################################################
# test数据集处理
data_transform_test = data_transform()

# 读取json文件
data_test = data_transform_test.read_data(path='./data/data_test.json')

# 提取需要信息
data_transform_test.extract_data(name='fact')
test_fact = data_transform_test.extraction['fact']

# #提取需要信息(分词速度太慢，保存分词结果)
# test_fact_cut=[jieba.lcut(i) for i in test_fact]

with open('./data_cut/test_fact_cut.json', 'r') as f:
    test_fact_cut = json.load(f)

test_fact_pad_seq = data_transform_test.text2seq(texts=test_fact_cut, needcut=False,
                                                 tokenizer_fact=tokenizer_fact, num_words=10000, maxlen=400)
np.save('./data_cut/test_fact_pad_seq.npy', test_fact_pad_seq)
# test_fact_pad_seq=np.load('./data_cut/test_fact_pad_seq.npy')
data_transform_test.extract_data(name='accusation')
test_accusations = data_transform_test.extraction['accusation']
data_transform_test.creat_label_set(name='accusation')
data_transform_test.creat_labels(name='accusation')
test_labels = np.array(data_transform_test.labels_one_hot)
np.save('./data_cut/test_labels.npy', test_labels)
