# Competiton_CAIL
![比赛图标](https://github.com/renjunxiang/Competiton_CAIL/blob/master/picture/比赛图标.png)<br>

## 比赛简介
为了促进法律智能相关技术的发展，在最高人民法院信息中心、共青团中央青年发展部的指导下，中国司法大数据研究院、中国中文信息学会、中电科系统团委联合清华大学、北京大学、中国科学院软件研究所共同举办“2018中国‘法研杯’法律智能挑战赛（CAIL2018）”。<br>
## 比赛任务：
* 罪名预测：根据刑事法律文书中的案情描述和事实部分，预测被告人被判的罪名；<br>
* 法条推荐：根据刑事法律文书中的案情描述和事实部分，预测本案涉及的相关法条；<br>
* 刑期预测：根据刑事法律文书中的案情描述和事实部分，预测被告人的刑期长短。<br>
## 成果说明：
代码分三大块，分别是：
### 数据预处理
* 方法模块data_transform.py和脚本data_preprocessing.py，包含读取数据文件、数据内容提取、分词、转成序号列表、文本长度统一<br>
* 受限于数据大小，仅提供验证集预处理后的数据供参考
``` python
from data_transform import data_transform
import json
import numpy as np

num_words=20000
maxlen=400

# train数据集处理
data_transform_train = data_transform()

# 读取json文件
data_train = data_transform_train.read_data(path='./data/data_train.json')

# 提取需要信息
data_transform_train.extract_data(name='fact')
train_fact = data_transform_train.extraction['fact']

#分词并保存原始分词结果，词语长度后期可以再改
train_fact_cut=data_transform_train.cut_texts(texts=train_fact,word_len=1,need_cut=True,
                               texts_cut_savepath='./data_deal/data_cut/train_fact_cut.json')

#抽取长度大于1的词语,目的在于去除标点和无意义词
train_fact_cut_new=data_transform_train.cut_texts(texts=train_fact_cut,word_len=2,need_cut=False,
                               texts_cut_savepath='./data_deal/data_cut/train_fact_cut_new.json')
							   
# 文本转序列
data_transform_train.text2seq(texts_cut=train_fact_cut_new, num_words=num_words, maxlen=maxlen)
tokenizer_fact = data_transform_train.tokenizer_fact

# 创建数据one-hot标签
data_transform_train.extract_data(name='accusation')
train_accusations = data_transform_train.extraction['accusation']
data_transform_train.creat_label_set(name='accusation')
train_labels = data_transform_train.creat_labels(name='accusation')
np.save('./data_deal/labels/train_labels_accusation.npy', train_labels)
```
### 模型训练
* 脚本model.py通过构建双向GRU网络，做多目标检测，结果在result文件夹内<br>
* 检查了预测结果，仅从文本数据的角度上看可能罪名标签还是有一些问题(不是质疑判决)，例如**valid数据中第十条**<br>
"criminals": ["连某某"], "term_of_imprisonment": {"death_penalty": false, "imprisonment": 14, "life_imprisonment": false}, "punish_of_money": 10000, "accusation": ["盗窃"], "relevant_articles": [264]}, "fact": "经审理查明：一、2016年8月16日早上，被告人连某某在保定市满城区精灵网吧二楼包厢内盗窃被害人李某某OPPOR7S手机一部。所盗手机已销售，赃款已挥霍。经保定市满城区涉案物品价格鉴证中心认定，被盗手机价值1982元。上述事实，被告人连某某在开庭审理过程中亦无异议，且有被害人李某某的陈述，涉案物品价格鉴证结论书，辨认笔录及照片，破案经过等证据证实，足以认定。二、2016年6月份一条下午，被告人连某某在保定市满城区现代网吧东侧一平房内盗窃被害人张某现金300元。所盗赃款已挥霍。上述事实，被告人连某某在开庭审理过程中亦无异议，且有被害人张某的陈述，辨认笔录及照片，破案经过等证据证实，足以认定。三、2016年8月22日晚，被告人连某某在保定市竞秀区秀兰城市美居小区北门口外孟某某经营的“爱车靓装”洗车店内盗窃行车记录仪两个及被害人李某经现金1700余元。行车记录仪被其扔掉，所盗赃款已挥霍。经保定市满城区涉案物品价格鉴证中心认定，被盗行车记录仪价值共计556元。上述事实，被告人连某某在开庭审理过程中亦无异议，且有被害人孟某某、李某经的陈述，涉案物品价格鉴证结论书，辨认笔录及照片，破案经过等证据证实，足以认定。四、2016年9月3日凌晨，被告人连某某在保定市千色网吧内盗窃被害人甄某某VIVOX6手机一部。所盗手机已销售，赃款已挥霍。经保定市满城区涉案物品价格鉴证中心认鉴定，被盗手机价值2202元。上述事实，被告人连某某在开庭审理过程中亦无异议，且有被害人甄某某的陈述，涉案物品价格鉴证结论书，辨认笔录及照片，破案经过等证据证实，足以认定。 **五、2015年9月23日下午，被告人连某某从保定市满城区野里村连某家里骗走金娃牌柴油三轮车一辆卖掉，所骗取三轮车已销售，赃款已挥霍。2016年9月份左右一天下午，连某某从保定市满城区野里村连某甲经营的石板厂骗走金娃牌柴油三轮车一辆卖掉，所骗取三轮车已销售，赃款已挥霍。2016年8月份左右一天下午，连某某从保定市满城区白堡村曹某某经营的鸡场骗走爱玛牌电动车一辆卖掉，所骗取电动车已销售，赃款已挥霍。** 经保定市满城区涉案物品价格鉴证中心认定，被盗两辆三轮车、一辆电动车价值分别为1700元、1500元、1250元。连某某父母已赔偿连某、连某甲、曹某某的经济损失。上述事实，被告人连某某在开庭审理过程中亦无异议，且有证人孙某某、连某某的证言，被害人连某、连某甲、曹某某的陈述，涉案物品价格鉴证结论书，辨认笔录及照片，破案经过等证据证实，足以认定。"} <br>
该论述出现的粗体属于诈骗，我的预测结果应该是正确的 **['诈骗', '盗窃']** ，但是数据集中仅有 **['盗窃']** 。

![](https://github.com/renjunxiang/Competiton_CAIL/blob/master/picture/Bidirectional_GRU_GlobalMaxPool1D_epochs.png)<br>
### 预测模块
* 按照大赛要求将分词、转成序号列表、文本长度统一和模型预测等步骤封装在predictor文件夹中<br>
``` python
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Bidirectional, GlobalMaxPool1D, Dropout
from keras.utils.vis_utils import plot_model
import numpy as np
import pandas as pd
from keras.models import load_model

num_words=20000
maxlen=400

train_fact_pad_seq = np.load('./data_deal/fact_pad_seq/train_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))
valid_fact_pad_seq = np.load('./data_deal/fact_pad_seq/valid_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))

train_labels = np.load('./data_deal/labels/train_labels_accusation.npy')
valid_labels = np.load('./data_deal/labels/valid_labels_accusation.npy')

set_accusation = np.load('./data_deal/set/set_accusation.npy')

model=load_model('./model/Bidirectional_GRU_GlobalMaxPool1D_epochs_2.h5')

def label2tag(labels):
    return [set_accusation[i == 1] for i in labels]


def predict2tag(predictions, n, score=True):
    if score:
        return [set_accusation[np.in1d(i, sorted(i, reverse=True)[0:n])] for i in predictions], \
               [i[np.in1d(i, sorted(i, reverse=True)[0:n])] for i in predictions]
    else:
        return [set_accusation[np.in1d(i, sorted(i, reverse=True)[0:n])] for i in predictions]

y=model.predict(valid_fact_pad_seq[:])
y1=label2tag(valid_labels[:])
y2=predict2tag(predictions=y,n=1, score=False)
y3=[set_accusation[i>0.5] for i in y]

# 只取最高置信度的准确率，训练2个epoch准确率为0.72，2个epoch准确率为0.78
p=[y1[i][0]==y2[i][0] for i in range(len(y1))]
print(sum(p)/len(p))
# 只取置信度大于0.5的准确率，训练2个epoch准确率为0.66，2个epoch准确率为0.76
q=[str(y1[i])==str(y2[i]) for i in range(len(y1))]
print(sum(q)/len(q))

r=pd.DataFrame({'label':y1,'predict':y2,'predict_list':y3})
r.to_excel('./result/valid_Bidirectional_GRU_epochs_2.xlsx',sheet_name='1',index=False)
```







