import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
import datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import torch
from torch import nn
from transformers import BertTokenizerFast as BertTokenizer, BertModel, get_linear_schedule_with_warmup, AdamW

from train_util import *
from pseudo_labeling import *
from model_setup import *

SEED = 123
BATCH_SIZE = 128
EPOCH_NUM = 3
WARMUP_STEPS = 10 ** 3
T1 = 100
T2 = 500
af = 3

TRAIN_PATH = 'train.csv'
TEST_PATH = '.test.csv'
comment_train = pd.read_csv(TRAIN_PATH)
comment_test = pd.read_csv(TEST_PATH)

comment_train = comment_train[['comment_text', 'toxic']].dropna().reset_index(drop=True)
comment_test = comment_test[['comment_text']].dropna().reset_index(drop=True)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
print('Current device is:', device)

BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)


comment_train = shuffle(comment_train, random_state=SEED)
val = comment_train[int(len(comment_train) * 0.8):].reset_index(drop=True)
train = comment_train[:int(len(comment_train) * 0.8)].reset_index(drop=True)
test = comment_test.copy()

trainSet = Data(tokenizer, 'train')
valSet = Data(tokenizer, 'val')
testSet = Data(tokenizer, 'test')

collate_fn = partial(collate_fn, device=device)
collate_fn_test = partial(collate_fn_test, device=device)
train_loader = construct_loader(trainSet, BATCH_SIZE, collate_fn)
val_loader = construct_loader(valSet, BATCH_SIZE, collate_fn)
test_loader = construct_loader(testSet, BATCH_SIZE, collate_fn_test)

model = BertClassifier(bert_model).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
total_steps = len(train_loader) * EPOCH_NUM - WARMUP_STEPS
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
loss_fn = nn.BCELoss()

for i in range(EPOCH_NUM):
    print(f"EPOCH {i+1}:")
    train(model, train_loader, optimizer, loss_fn, scheduler)
    evaluate(model, val_loader)

pseudo_labeling(model, train_loader, test_loader, optimizer, loss_fn)

torch.save(model.state_dict(), 'toxic_classifier.pt')



