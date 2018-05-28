# 预测模块
* 按照大赛要求将分词、转成序号列表、文本长度统一和模型预测等步骤封装在predictor文件夹中<br>
* 文本转序列的脚本为data_transform.py
* 由于上传速度太慢，全部代码包括读取、清洗等均在github上，地址<https://github.com/renjunxiang/Competition_CAIL>
``` python
from predictor import Predictor

content = ['菏泽市牡丹区人民检察院指控，被告人侯某于2014年10月14日晨2时许，'
           '在菏泽市牡丹区万福办事处赵庄社区罗马皇宫KTV，因琐事对点歌员张某实施殴打，'
           '致张某双侧鼻骨骨折、上颌骨额突骨折，经法医鉴定，被害人张某的损伤程度为轻伤二级。']
model = Predictor()
p = model.predict(content)
print(p)
``` 