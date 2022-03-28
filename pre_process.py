from random import sample
import json
import pandas as pd
import numpy as np
import os

BASEPATH = os.path.dirname(__file__)
DOCPATH = os.path.join(BASEPATH, './raw_data/test_data')
CATNAMEPATH = os.path.join(BASEPATH, './processed_data/catagories_name.json')
TESTPATH = os.path.join(BASEPATH, './processed_data/test_data.csv')
docid = range(1, 11)

all_df = pd.DataFrame(columns=['docid', 'sentence', 'category'])

for file_i in docid:
    file_name = os.path.join(DOCPATH, f'd{file_i}.txt')
    file_df = pd.read_csv(file_name, sep='\t',
                          engine='python', names=['sentence', 'category'], quoting=3)
    file_df.insert(0, 'docid', file_i)
    # file_df.info()
    all_df = all_df.append(file_df, ignore_index=True)

print(all_df.info())

with open(CATNAMEPATH, newline='') as f:
    labels = list(json.load(f).values())
    # print(labels)

# change label to integer
all_df['category'] = all_df['category'].apply(lambda x: labels.index(x))
print(all_df.head())

all_df.to_csv(TESTPATH, index=False)

# train_idxs = sample(docid, int(len(docid)*0.8))
# train_df = all_df.loc[all_df['docid'].isin(train_idxs)]
# valid_df = all_df.loc[~all_df['docid'].isin(train_idxs)]
# print(train_df.head())
# print(valid_df.head())
# valid_df.to_csv("./processed_data/valid_data.csv", index=False)
# train_df.to_csv("./processed_data/train_data.csv", index=False)

