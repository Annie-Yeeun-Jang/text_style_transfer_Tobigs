# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartForSequenceClassification 
from transformers import Trainer, TrainingArguments

sys.path.append("")
from utils.helper import evaluate_sc
from utils.dataset import SCIterator
from utils.dataset import load_embedding
from utils.optim import ScheduledOptim

import pandas as pd
import random
from sklearn.metrics import accuracy_score

# load_config 불러오기 추가
import yaml
from utils.helper import load_config

class DataSet(Dataset):
    def __init__(self, source, target, tokenizer, input_length, output_length):         
        self.source=source
        self.target = target
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
  
    def __len__(self):
        return len(self.target)
    
    # def clean_text(self, text):
    #     text = text.replace('~', '')
    #     return text
      
    def __getitem__(self, index):
        source_i = self.tokenizer.encode_plus(self.source[index], max_length=self.output_length, 
                                                padding='max_length', truncation=True, return_tensors="pt")
        # target_i = torch.tensor([self.target[index][0]])

        target_i = torch.tensor([self.target[index]])
        source_ids = source_i["input_ids"].squeeze()
        src_mask    = source_i["attention_mask"].squeeze()

        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_i}


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions[0].argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }
 
def main():

    device = 'cuda' if cuda.is_available() else 'cpu'

    print('[Info] device is {}'.format(device))

    # Load config dictionary
    config_path = "./base.yaml"
    cfg = load_config(config_path)
    opt = cfg['classifier']

    torch.manual_seed(opt['seed'])


    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # for token in ['<E>', '<F>']:
    #     tokenizer.add_tokens(token)


    train_formal = pd.read_csv("./dataset/train.1",sep='\t',header=None)
    train_informal = pd.read_csv("./dataset/train.0",sep='\t',header=None)
    test_formal = pd.read_csv("./dataset/test.1",sep='\t',header=None)
    test_informal = pd.read_csv("./dataset/test.0",sep='\t',header=None)

    train_formal['label'] = 1
    train_informal['label']=0
    train_data = pd.concat([train_formal,train_informal])
    train_data = train_data.sample(frac=1) 
    test_formal['label'] = 1
    test_informal['label']=0
    test_data = pd.concat([test_formal,test_informal])
    test_data = test_data.sample(frac=1) 
    train = list(train_data[0])
    train_label= list(train_data['label'])
    test=list(test_data[0])
    test_label =list(test_data['label'])

    train_dataset = DataSet(train,train_label, tokenizer, 16, 16)
    test_dataset = DataSet(test,test_label, tokenizer, 16, 16)



    model = BartForSequenceClassification.from_pretrained('facebook/bart-large').to(device)

    training_args = TrainingArguments(
        output_dir='./classifiers',          # output directory
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=200,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        # logging_dir='./logs',            # directory for storing logs
        load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=500,               # log & save weights each logging_steps
        evaluation_strategy="steps",     # evaluate each `logging_steps`
        # learning_rate= 1e-4,
    )
    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,          # evaluation dataset
        compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )

    print('[Info] Built a model with {} parameters'.format(
           sum(p.numel() for p in model.parameters())))
    print('[Info] Training Start')

    # train the model
    trainer.train()
    torch.save(model, cfg['model_path']+'BART'+'classifier.pt')
    

if __name__ == '__main__':
    main()
