from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split

num_words = 80000
maxlen = 400
kernel_size = 3
DIM = 512
batch_size = 256

fact = np.load('./data_deal/big_fact_pad_seq_%d_%d.npy' % (num_words, maxlen))
fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)
del fact

fact_train_shuffle = deepcopy(fact_train)
for i in fact_train_shuffle:
    np.random.shuffle(i)
np.save('./data_deal/big_fact_pad_seq_shuffle_%d_%d.npy' % (num_words, maxlen), fact_train_shuffle)
