import json

with open('./data_deal/data_cut/train_fact_cut.json', 'r') as f:
    train_fact_cut = json.load(f)

train_fact_cut_new = []
for n, i in enumerate(train_fact_cut):
    train_fact_cut_new.append([j for j in i if len(j) > 1])
    if n % 1000 == 0:
        print(n)

with open('./data_deal/data_cut/train_fact_cut_new.json','w') as f:
    json.dump(train_fact_cut_new,f)
######################################################################
with open('./data_deal/data_cut/valid_fact_cut.json', 'r') as f:
    valid_fact_cut = json.load(f)

valid_fact_cut_new = []
for n, i in enumerate(valid_fact_cut):
    valid_fact_cut_new.append([j for j in i if len(j) > 1])
    if n % 1000 == 0:
        print(n)

with open('./data_deal/data_cut/valid_fact_cut_new.json','w') as f:
    json.dump(valid_fact_cut_new,f)
######################################################################
with open('./data_deal/data_cut/test_fact_cut.json', 'r') as f:
    test_fact_cut = json.load(f)

test_fact_cut_new = []
for n, i in enumerate(test_fact_cut):
    test_fact_cut_new.append([j for j in i if len(j) > 1])
    if n % 1000 == 0:
        print(n)

with open('./data_deal/data_cut/test_fact_cut_new.json','w') as f:
    json.dump(test_fact_cut_new,f)