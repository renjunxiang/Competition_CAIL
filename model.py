from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, initializers, Dropout, Masking
from keras.layers import Conv1D, InputLayer
from keras.layers.recurrent import LSTM,GRU
from keras.optimizers import SGD, Adagrad, Adam
import numpy as np

n = 0
tokenizer_fact = Tokenizer(num_words=5000)
while n < len(train_fact_cut):
    tokenizer_fact.fit_on_texts(texts=train_fact_cut[n:n + 10000])
    n += 10000
    print('finish fit %d samples' % n)
fact_seq = tokenizer_fact.texts_to_sequences(texts=train_fact_cut)
fact_pad_seq = pad_sequences(fact_seq[0:100], maxlen=500, padding='post', value=0, dtype='int')

fact_pad_seq=[]
n = 0
while n < len(train_fact_cut):
    fact_pad_seq+=list(pad_sequences(fact_seq[n:n+10000], maxlen=500, padding='post', value=0, dtype='int'))
    n += 10000
    print('finish fit %d samples' % n)

m=np.array(fact_pad_seq)
np.save()



c=np.concatenate((a,b),axis=0)