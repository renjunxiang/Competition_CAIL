import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Embedding, Input,Dropout
from keras.layers import BatchNormalization, Concatenate
import pandas as pd
import time
from keras.models import load_model
from evaluate import predict2both, predict2half, predict2top, f1_avg
from textcnn import textcnn_one

print('start', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
print('accusation')

num_words = 80000
maxlen = 400
filters = 256
print('num_words = 80000, maxlen = 400')

# fact数据集
fact = np.load('./data_deal/big_fact_pad_seq_%d_%d.npy' % (num_words, maxlen))
fact_train, fact_test = train_test_split(fact, test_size=0.05, random_state=1)
del fact

# 标签数据集
labels = np.load('./data_deal/labels/big_labels_accusation.npy')
labels_train, labels_test = train_test_split(labels, test_size=0.05, random_state=1)
del labels

set_accusation = np.load('./data_deal/set/set_accusation.npy')

data_input = Input(shape=[maxlen])
word_vec = Embedding(input_dim=num_words + 1,
                     input_length=maxlen,
                     output_dim=512,
                     mask_zero=False,
                     name='Embedding')(data_input)

x1 = textcnn_one(word_vec=word_vec, kernel_size=1, filters=filters)
x2 = textcnn_one(word_vec=word_vec, kernel_size=2, filters=filters)
x3 = textcnn_one(word_vec=word_vec, kernel_size=3, filters=filters)
x4 = textcnn_one(word_vec=word_vec, kernel_size=4, filters=filters)
x5 = textcnn_one(word_vec=word_vec, kernel_size=5, filters=filters)

x = Concatenate(axis=1)([x1, x2, x3, x4, x5])
x = BatchNormalization()(x)
x = Dense(1000, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(labels_train.shape[1], activation="sigmoid")(x)
model = Model(inputs=data_input, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

n_start = 1
n_end = 21
score_list1 = []
score_list2 = []
for i in range(n_start, n_end):
    model.fit(x=fact_train, y=labels_train, batch_size=512, epochs=1, verbose=1)

    model.save('./model/%d_%d/accusation/TextCNN_%d_epochs_%d.h5' % (num_words, maxlen, filters, i))

    y = model.predict(fact_test[:])
    y1 = predict2top(y)
    y2 = predict2half(y)
    y3 = predict2both(y)

    print('%s accu:' % i)
    # 只取最高置信度的准确率
    s1 = [(labels_test[i] == y1[i]).min() for i in range(len(y1))]
    print(sum(s1) / len(s1))
    # 只取置信度大于0.5的准确率
    s2 = [(labels_test[i] == y2[i]).min() for i in range(len(y1))]
    print(sum(s2) / len(s2))
    # 结合前两个
    s3 = [(labels_test[i] == y3[i]).min() for i in range(len(y1))]
    print(sum(s3) / len(s3))

    print('%s f1:' % i)
    # 只取最高置信度的准确率
    s4 = f1_avg(y_pred=y1, y_true=labels_test)
    print(s4)
    # 只取置信度大于0.5的准确率
    s5 = f1_avg(y_pred=y2, y_true=labels_test)
    print(s5)
    # 结合前两个
    s6 = f1_avg(y_pred=y3, y_true=labels_test)
    print(s6)

    score_list1.append([i,
                        sum(s1) / len(s1),
                        sum(s2) / len(s2),
                        sum(s3) / len(s3)])
    score_list2.append([i, s4, s5, s6])

print(pd.DataFrame(score_list1))
print(pd.DataFrame(score_list2))
print('end', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
print('#####################\n')
# nohup python model_CNN_accusation.py 2>&1 &
