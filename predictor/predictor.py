import pickle
from data_transform import data_transform


class Predictor:
    def __init__(self, num_words=20000, maxlen=400,
                 model=None,
                 tokenizer_path='D:/work/svn/CAIL/data_deal/models/tokenizer_fact_20000.pkl'):
        self.num_words = num_words
        self.maxlen = maxlen
        self.model=model
        self.batch_size = 128
        self.content_transform = data_transform()
        self.tokenizer_path=tokenizer_path

    def predict(self, content):
        num_words = self.num_words
        maxlen = self.maxlen
        model=self.model
        content_transform = self.content_transform
        tokenizer_path=self.tokenizer_path
        # 分词
        content_cut = content_transform.cut_texts(texts=content, word_len=2)
        with open(tokenizer_path, mode='rb') as f:
            tokenizer_fact = pickle.load(f)
        content_transform.text2seq(texts_cut=content_cut, tokenizer_fact=tokenizer_fact,
                                   num_words=num_words, maxlen=maxlen)
        content_fact_pad_seq = content_transform.fact_pad_seq
        result = []
        for a in range(0, len(content)):
            result.append({
                "accusation": [1, 2, 3],
                "imprisonment": 5,
                "articles": [5, 7, 9]
            })
        return content_fact_pad_seq

if __name__ == '__main__':
    predictor=Predictor()
    content=['我爱北京天安门', '收款方哪家口碑北京开始，数据库备份围绕健康上网电费']
    m=predictor.predict(content,
                        tokenizer_path='D:/work/svn/CAIL/data_deal/models/tokenizer_fact_20000.pkl')
    print(m)
    # from data_transform import data_transform
    # content_transform = data_transform()
    # content_transform.texts_cut(texts=content, word_len=2)

