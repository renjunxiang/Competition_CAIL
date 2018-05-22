import pickle
from keras.models import load_model
from data_transform import data_transform
import numpy as np


class Predictor:
    def __init__(self, num_words=20000, maxlen=400,
                 model_path='./model/Bidirectional_GRU_GlobalMaxPool1D_epochs_2.h5',
                 tokenizer_path='./model/tokenizer_fact_20000.pkl'):
        self.num_words = num_words
        self.maxlen = maxlen
        self.model_path=model_path
        self.batch_size = 128
        self.content_transform = data_transform()
        self.tokenizer_path=tokenizer_path

    def predict(self, content):
        num_words = self.num_words
        maxlen = self.maxlen
        model_path=self.model_path
        content_transform = self.content_transform
        tokenizer_path=self.tokenizer_path
        # 分词
        content_cut = content_transform.cut_texts(texts=content, word_len=2)
        with open(tokenizer_path, mode='rb') as f:
            tokenizer_fact = pickle.load(f)
        content_transform.text2seq(texts_cut=content_cut, tokenizer_fact=tokenizer_fact,
                                   num_words=num_words, maxlen=maxlen)
        content_fact_pad_seq = np.array(content_transform.fact_pad_seq)
        model=load_model(model_path)
        accusation=model.predict(content_fact_pad_seq)
        n=accusation.shape[1]
        def transform(x):
            score_max=x.max()
            return [i for i in range(n) if x[i] == score_max]
        result = []
        for a in range(0, len(content)):
            result.append({
                "accusation": transform(accusation[a]),
                "imprisonment": None,
                "articles": [None]
            })
        return result

if __name__ == '__main__':
    content=['我爱北京天安门', '收款方哪家口碑北京开始，数据库备份围绕健康上网电费']
    predictor=Predictor()
    m=predictor.predict(content)
    print(m)
    # from data_transform import data_transform
    # content_transform = data_transform()
    # content_transform.texts_cut(texts=content, word_len=2)

