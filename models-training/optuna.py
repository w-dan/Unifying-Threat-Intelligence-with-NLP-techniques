# -*- coding: utf-8 -*-
"""bert_OPTUNA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/w-dan/MATM/blob/main/MATM-code/data-analysis/bert_OPTUNA.ipynb

# Initial setup
"""

# core
import os
from dotenv import load_dotenv
import numpy as np

# dataset
from datasets import Dataset
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split


# training
import torch
from transformers import(
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
import optuna

# evaluation
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
import pickle, json

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch

from bert_utils import *

load_dotenv(".env")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
TOKEN = os.getenv("HUGGINGFACE_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

collection_name = "dataset"
DATABASE_NAME = "APTs"

"""# Dataset preparation"""

df = fetch_and_preprocess_data(DATABASE_NAME, collection_name, CONNECTION_STRING, preprocess=True, field_to_get="tactics", include_tactics=True)
df

df_one_hot_encoded = process_tactics(df)
df_one_hot_encoded

tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

texts = df_one_hot_encoded['corpus'].tolist()
labels = df_one_hot_encoded.drop(columns=['corpus', 'tactics_length']).values

def convert_to_float32(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.array(x, dtype=np.float32)
    else:
        return np.float32(x)

labels_list = list(labels)

# making sure they are all the same length
max_length = max(len(label) for label in labels)
for i in range(len(labels)):
    if len(labels[i]) < max_length:
        labels[i] = np.pad(labels[i], (0, max_length - len(labels[i])), 'constant')

labels = np.vstack(labels)  # stacking the labels into a 2D array



print(len(texts))
print(len(labels))

train_dataset, val_dataset, test_dataset = prepare_datasets(texts, labels, tokenizer)
print(type(train_dataset))

"""# Model and training"""



num_labels = labels.shape[1]
model = RobertaForSequenceClassification.from_pretrained(
    'FacebookAI/roberta-base',
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

data_collator = DataCollatorWithPadding(tokenizer)

def objective(trial):
    # hyperparameters to optimize
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    training_args = TrainingArguments(
        output_dir=f'./results/{trial.number}',  # output directory
        num_train_epochs=num_train_epochs,       # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,            # batch size for evaluation
        warmup_steps=500,                        # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,               # strength of weight decay (formerly 0.01)
        logging_dir=f'./logs/{trial.number}',    # directory for storing logs
        logging_steps=10,
        learning_rate=learning_rate,
    )

    # model and trainer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=max_length, problem_type="multi_label_classification")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )


    # saving the training logs along with hyperparameters
    training_logs = trainer.train()
    log_data = {
        'trial_number': trial.number,
        'num_train_epochs': num_train_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'training_logs': training_logs
    }

    with open(f'./results/{trial.number}/training_logs.pkl', 'wb') as f:
        pickle.dump(log_data, f)

    eval_result = trainer.evaluate()

    # saving the evaluation results
    with open(f'./results/{trial.number}/eval_results.json', 'w') as f:
        json.dump(eval_result, f)

    return eval_result['eval_loss']

"""Optuna search:"""

print(train_dataset.labels.shape)
print(test_dataset.labels.shape)
print(val_dataset.labels.shape)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# saving best hyperparameters
best_hyperparams = study.best_params
with open('./results/best_hyperparams.json', 'w') as f:
    json.dump(best_hyperparams, f)

print(f"Best Hyperparameters: {best_hyperparams}")
