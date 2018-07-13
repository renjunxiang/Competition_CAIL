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

# 创建数据one-hot标签
data_transform_big.extract_data(name='accusation')
big_accusations = data_transform_big.extraction['accusation']
data_transform_big.creat_label_set(name='accusation')
big_labels = data_transform_big.creat_labels(name='accusation')
np.save('./data_deal/labels/big_labels_accusation.npy', big_labels)

# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')
data_transform_big.extract_data(name='relevant_articles')
big_relevant_articless = data_transform_big.extraction['relevant_articles']
data_transform_big.creat_label_set(name='relevant_articles')
big_labels = data_transform_big.creat_labels(name='relevant_articles')
np.save('./data_deal/labels/big_labels_relevant_articles.npy', big_labels)

# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')

# 创建刑期连续变量
data_transform_big.extract_data(name='imprisonment')
big_imprisonments = data_transform_big.extraction['imprisonment']
np.save('./data_deal/labels/big_labels_imprisonments.npy', big_imprisonments)

# big数据集处理
data_transform_big = data_transform()

# 读取json文件,1710857行
data_transform_big.read_data(path='./data/cail2018_big.json')

# 创建刑期离散变量
data_transform_big.extract_data(name='imprisonment')
big_imprisonments = data_transform_big.extraction['imprisonment']
data_transform_big.creat_label_set(name='imprisonment')
big_labels = data_transform_big.creat_labels(name='imprisonment')
np.save('./data_deal/labels/big_labels_imprisonments_discrete.npy', big_labels)

