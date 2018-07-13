import pickle
import jieba
import json
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

jieba.setLogLevel('WARN')

num_words = 80000
maxlen = 400

tokenizer_fact = Tokenizer(num_words=num_words)

# train tokenizer
# for i in range(18):
#     print('start big_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
#     with open('./data_deal/data_cut/big_fact_cut_%d_%d_new.pkl' % (i * 100000, i * 100000 + 100000), mode='rb') as f:
#         big_fact_cut = pickle.load(f)
#     texts_cut_len = len(big_fact_cut)
#     n = 0
#     # 分批训练
#     while n < texts_cut_len:
#         tokenizer_fact.fit_on_texts(texts=big_fact_cut[n:n + 10000])
#         n += 10000
#         if n < texts_cut_len:
#             print('tokenizer finish fit %d samples' % n)
#         else:
#             print('tokenizer finish fit %d samples' % texts_cut_len)
#     print('finish big_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
#
# with open('./model/tokenizer_fact_%d.pkl' % (num_words), mode='wb') as f:
#     pickle.dump(tokenizer_fact, f)

with open('./model/tokenizer_fact_%d.pkl' % (num_words), mode='rb') as f:
    tokenizer_fact=pickle.load(f)

# texts_to_sequences
for i in range(18):
    print('start big_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
    with open('./data_deal/data_cut/big_fact_cut_%d_%d_new.pkl' % (i * 100000, i * 100000 + 100000), mode='rb') as f:
        big_fact_cut = pickle.load(f)
    # 分批执行 texts_to_sequences
    big_fact_seq = tokenizer_fact.texts_to_sequences(texts=big_fact_cut)
    with open('./data_deal/fact_seq/fact_seq_%d_%d.pkl' % (i * 100000, i * 100000 + 100000), mode='wb') as f:
        pickle.dump(big_fact_seq, f)
    print('finish big_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))

# pad_sequences
for i in range(18):
    print('start big_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
    with open('./data_deal/fact_seq/fact_seq_%d_%d.pkl' % (i * 100000, i * 100000 + 100000), mode='rb') as f:
        big_fact_seq = pickle.load(f)
    texts_cut_len = len(big_fact_seq)
    n = 0
    fact_pad_seq = []
    # 分批执行pad_sequences
    while n < texts_cut_len:
        fact_pad_seq += list(pad_sequences(big_fact_seq[n:n + 20000], maxlen=maxlen,
                                           padding='post', value=0, dtype='int'))
        n += 20000
        if n < texts_cut_len:
            print('finish pad_sequences %d samples' % n)
        else:
            print('finish pad_sequences %d samples' % texts_cut_len)
    with open('./data_deal/fact_pad_seq/fact_pad_seq_%d_%d_%d.pkl' % (maxlen, i * 100000, i * 100000 + 100000),
              mode='wb') as f:
        pickle.dump(fact_pad_seq, f)

# 汇总pad_sequences,5G,16G内存够用
maxlen = 400
num_words = 40000
fact_pad_seq = []
for i in range(18):
    print('start big_fact_cut_%d_%d' % (i * 100000, i * 100000 + 100000))
    with open('./data_deal/fact_pad_seq/fact_pad_seq_%d_%d_%d.pkl' % (maxlen, i * 100000, i * 100000 + 100000),
              mode='rb') as f:
        fact_pad_seq += pickle.load(f)
fact_pad_seq = np.array(fact_pad_seq)
np.save('./data_deal/fact_pad_seq/big_fact_pad_seq_%d_%d.npy' % (num_words, maxlen), fact_pad_seq)