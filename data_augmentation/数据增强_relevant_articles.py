import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
path = '/home/admini/data/competition/CAIL/data/cail2018_big.json'
f = open(path, 'r', encoding='utf8')
line = True
data = []
n = 0
while line:
    line = f.readline()
    try:
        data.append(json.loads(line))
    except Exception as e:
        print('num: %d' % n)
        print('error: %s' % e)
        print('data: %s' % line)
        print(n)
    n += 1
    if n % 200000 == 0:
        print('finish read %d lines' % n)

data_train, data_test = train_test_split(data, test_size=0.05, random_state=1)


class_name = []
class_index = []#1442617

for n, i in enumerate(data_train):
    if len(i['meta']['relevant_articles']) == 1:
        class_name.append(i['meta']['relevant_articles'][0])
        class_index.append(n)

# x=pd.Series(class_name)
# x.value_counts()

class_name_array = np.array(class_name)
class_index_array = np.array(class_index)
np.save('/home/admini/data/competition/CAIL/data_deal/class/name_relevant_articles.npy', class_name_array)
np.save('/home/admini/data/competition/CAIL/data_deal/class/index_relevant_articles.npy', class_index_array)

maxcount=100000
num=100
relevant_articles_set = list(set(class_name))
index_add = []
m = 0
for i in relevant_articles_set:
    class_count_i = class_name.count(i)
    class_index_i = class_index_array[class_name_array == i]
    print('class_count_i:',class_count_i)
    n = max(int(maxcount / class_count_i) - 1,0)
    print('n:',n)
    m += (min(num, n) * class_count_i)
    print(m)
    if n > 1:
        index_add += list(class_index_i) * min(num, n)
np.save('/home/admini/data/CAIL/data_deal/index_add_relevant_articles_%d_%d.npy'%(maxcount,num), np.array(index_add))

'''
133    467986
264    344406
234    175309
266     49267
354     47843
293     30086
347     27990
303     24841
263     16298
128     15958
345     13696
277     12389
348     12143
312     11638
196      9585
275      9354
267      7802
224      6916
271      6703
232      6509
141      6181
359      5749
274      5699
233      5541
115      5065
114      4772
280      4587
342      4464
144      4381
238      4161
'''
######################################################################################################
import numpy as np
from sklearn.model_selection import train_test_split

num_words = 80000
maxlen = 400
kernel_size = 3
DIM = 512
batch_size = 256

print('num_words = 80000, maxlen = 400')

# fact数据集
fact = np.load('./data_deal/big_fact_pad_seq_%d_%d.npy' % (num_words, maxlen))
fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)
del fact

# 标签数据集
labels = np.load('./data_deal/labels/big_labels_relevant_articles.npy')
labels_train, labels_test = train_test_split(labels, test_size=0.05, random_state=1)
del labels

# set_relevant_articles = np.load('./data_deal/set/set_relevant_articles.npy')
index_add_relevant_articles=np.load('./data_deal/index_add_relevant_articles.npy')
fact_train=np.concatenate([fact_train,fact_train[index_add_relevant_articles]],axis=0)
labels_train=np.concatenate([labels_train,labels_train[index_add_relevant_articles]],axis=0)


