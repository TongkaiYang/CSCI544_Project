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

class Data(Dataset):
    def __init__(self, tokenizer, data_type):
        self.data_type = data_type
        if data_type == 'train':
            self.data = train
        elif data_type == 'val':
            self.data = val
        elif data_type == 'test':
            self.data = test
        self.X, self.Y = [], []
        for i, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            if self.data_type != 'test':
                x, y = self.row_to_tensor(tokenizer, row)
                self.X.append(x)
                self.Y.append(y)
            else:
                self.X.append(self.row_to_tensor(tokenizer, row))

    def row_to_tensor(self, tokenizer, row):
        tokens = tokenizer.encode(row['comment_text'], add_special_tokens=True)
        if len(tokens) > 120:
            tokens = tokens[:119] + [tokens[-1]]
        x = torch.LongTensor(tokens)
        if self.data_type != 'test':
            y = torch.FloatTensor([row['toxic']])
            return x, y
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data_type != 'test':
            return self.X[index], self.Y[index]
        return self.X[index]

def collate_fn(batch, device):
    x, y = list(zip(*batch))
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x.to(device), y.to(device)

def collate_fn_test(batch, device):
    x = list(batch)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    return x.to(device)

def construct_loader(dataset, batch_size, collate_fn):
	return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
		)

class BertClassifier(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]
        cls_output = self.classifier(cls_output)
        cls_output = torch.sigmoid(cls_output)
        return cls_output