from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Bidirectional, GlobalMaxPool1D, Dropout
import numpy as np
import pandas as pd

num_words=20000
maxlen=400

train_fact_pad_seq = np.load('./data_deal/fact_pad_seq/train_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))
valid_fact_pad_seq = np.load('./data_deal/fact_pad_seq/valid_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))
test_fact_pad_seq = np.load('./data_deal/fact_pad_seq/test_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))

train_labels = np.load('./data_deal/labels/train_labels_accusation.npy')
valid_labels = np.load('./data_deal/labels/valid_labels_accusation.npy')
test_labels = np.load('./data_deal/labels/test_labels_accusation.npy')

set_accusation = np.load('./data_deal/set/set_accusation.npy')

data_input = Input(shape=[valid_fact_pad_seq.shape[1]])
word_vec = Embedding(input_dim=num_words + 1,
                     input_length=maxlen,
                     output_dim=128,
                     mask_zero=0,
                     name='Embedding')(data_input)
x = Bidirectional(GRU(500, return_sequences=True))(word_vec)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(500, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(valid_labels.shape[1], activation="sigmoid")(x)
model = Model(inputs=data_input, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=train_fact_pad_seq, y=train_labels,
          batch_size=100, epochs=1,
          validation_data=(valid_fact_pad_seq, valid_labels), verbose=1)

model.save('./model/Bidirectional_GRU_GlobalMaxPool1D_epochs_2.h5')

x = model.predict(valid_fact_pad_seq[0:100])


def label2tag(labels):
    return [set_accusation[i == 1] for i in labels]


def predict2tag(predictions, n, score=True):
    if score:
        return [set_accusation[np.in1d(i, sorted(i, reverse=True)[0:n])] for i in predictions], \
               [i[np.in1d(i, sorted(i, reverse=True)[0:n])] for i in predictions]
    else:
        return [set_accusation[np.in1d(i, sorted(i, reverse=True)[0:n])] for i in predictions]


print(label2tag(valid_labels[11500:11520]), '\n',
      predict2tag(predictions=model.predict(valid_fact_pad_seq[11500:11520]), n=1, score=False))

sum(x[0] > 0.5)
sorted(x[0], reverse=True)[0:3]


y1=label2tag(valid_labels[:])
y2=predict2tag(predictions=model.predict(valid_fact_pad_seq[:]),n=1, score=False)
r=pd.DataFrame({'label':y1,'predict':y2})
r.to_excel('./result/valid_Bidirectional_GRU_epochs_2.xlsx',sheet_name='1',index=False)
p=[y1[i][0]==y2[i][0] for i in range(len(y1))]
sum(p)/len(p)

with open('./data/%s.txt' % 'accu',encoding='utf-8') as f:
    label_set = f.readlines()
    label_set=[i[:-1] for i in label_set]


