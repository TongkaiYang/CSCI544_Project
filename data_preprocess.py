import pandas as pd
from tqdm.notebook import tqdm
from data_preprocessing_util import *

TRAIN_PATH = '/data/train1.csv'
TRAIN_PATH2 = '/data/train2.csv'
TRAIN_PATH3 = '/data/train3.csv'
TEST_PATH = '/data/test.csv'
comment_train1 = pd.read_csv(TRAIN_PATH)
comment_train2 = pd.read_csv(TRAIN_PATH2)
comment_train3 = pd.read_csv(TRAIN_PATH3)
comment_test = pd.read_csv(TEST_PATH)

comment_train1 = comment_train1[['comment_text', 'toxic']]
comment_train2['toxic'] = comment_train2['Insult']
comment_train2 = comment_train2.drop(['Insult', 'Date'], axis=1)
comment_train3['toxic'] = comment_train3['class'].apply(lambda x: 1 if x in (1, 0) else 0)
comment_train3 = comment_train3[['comment_text', 'toxic']]
comment_train = pd.concat([comment_train1, comment_train2, comment_train3])
comment_train = comment_train.dropna().reset_index(drop=True)
comment_test = comment_test.drop(['id'], axis=1)

print(
    'Training data shape:', comment_train.shape[0],
    '\n',
    'Testing data shape:', comment_test.shape[0]
 )

tqdm.pandas()
comment_train['comment_text'] = comment_train['comment_text'].str.lower().progress_apply(data_preprocessing)
comment_test['comment_text'] = comment_test['comment_text'].str.lower().progress_apply(data_preprocessing)

comment_train.to_csv('train.csv')
comment_test.to_csv('test.csv')