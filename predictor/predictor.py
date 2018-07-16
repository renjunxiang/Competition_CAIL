import pickle
from keras.models import load_model
from .data_transform import data_transform
import numpy as np
import os

localpath = os.path.dirname(__file__)


class Predictor:
    def __init__(self, num_words=80000, maxlen=400,
                 accusation_path=localpath + '/model/accusation/DIM_256_CNN_BN_gpool_epochs_19.h5',
                 relevant_articles_path=localpath + '/model/relevant_articles/DIM_512_RES_BN_gpool_bs_256_epochs_18.h5',
                 imprisonments_path=localpath + '/model/imprisonments/DIM_512_CNN_gpool_BN_epochs_10.h5',
                 tokenizer_path=localpath + '/model/tokenizer_fact_80000.pkl'):
        self.num_words = num_words
        self.maxlen = maxlen
        self.accusation_path = accusation_path
        self.relevant_articles_path = relevant_articles_path
        self.batch_size = 512
        self.content_transform = data_transform()
        self.tokenizer_path = tokenizer_path
        self.model1 = load_model(accusation_path)
        self.model2 = load_model(relevant_articles_path)
        self.model3 = load_model(imprisonments_path)

    def predict(self, content):
        num_words = self.num_words
        maxlen = self.maxlen
        content_transform = self.content_transform
        tokenizer_path = self.tokenizer_path
        # 分词
        content_cut = content_transform.cut_texts(texts=content, word_len=2)
        with open(tokenizer_path, mode='rb') as f:
            tokenizer_fact = pickle.load(f)
        content_transform.text2seq(texts_cut=content_cut, tokenizer_fact=tokenizer_fact,
                                   num_words=num_words, maxlen=maxlen)
        content_fact_pad_seq = np.array(content_transform.fact_pad_seq)

        model1 = self.model1
        accusation = model1.predict(content_fact_pad_seq)
        model2 = self.model2
        relevant_articles = model2.predict(content_fact_pad_seq)
        model3 = self.model3
        imprisonments = model3.predict(content_fact_pad_seq)

        def transform(x):
            n = len(x)
            x_return = np.arange(1, n + 1)[x > 0.5].tolist()
            if len(x_return) == 0:
                x_return = np.arange(1, n + 1)[x == x.max()].tolist()
            return x_return

        result = []
        for i in range(0, len(content)):
            if imprisonments[i][0] > 400:
                imprisonment = -2
            elif imprisonments[i][0] > 300:
                imprisonment = -1
            else:
                imprisonment = int(np.round(imprisonments[i][0], 0))

            result.append({
                "accusation": transform(accusation[i]),
                "articles": transform(relevant_articles[i]),
                "imprisonment": imprisonment
            })
        return result


if __name__ == '__main__':
    content = ['我爱北京天安门', '收款方哪家口碑北京开始，数据库备份围绕健康上网电费']
    predictor = Predictor()
    m = predictor.predict(content)
    print(m)
