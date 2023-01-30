# model
import copy
import json
import os
import pdb
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from tqdm import tqdm
import config_eval_crisis as config
import config_main

import function_crisismd as f


import requests
import pypac

def request(method, url, **kwargs):
    with pypac.PACSession() as session:
        return session.request(method=method, url=url, **kwargs)

requests.request = request


VERBOSE = True
PATH = config.CRISIS_MD_DIR 
num_labels = config.num_labels
batch_size= config.batch_size_crisismd
n_epochs = config.n_epochs
criterion = nn.CrossEntropyLoss()
labels_descirption = {
    0: 'not_informative',
    1: 'informative'
}
torch.manual_seed(1234)

def create_df():
    with open(PATH + '/results.json', 'r') as file:
        id_time = json.load(file)

    df = f.get_data(path=PATH)
    df = f.build_df(df=df, id_time=id_time)
    df.columns = ['id', 'event', 'period', 'l', 'label', 'intention_annotation', 'created_at', 'text']
    df = df[['id', 'event', 'text', 'created_at', 'label']].drop_duplicates()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df.label)
    return df

def eval(model_name="bloom"):
    device = config_main.device 
    df = create_df()
    
    # split
    val = df.iloc[config.splits['val']]
    test = df.iloc[config.splits['test']]
    #train = df.iloc[config.splits['train']]
    subset_train = random.sample(config.splits['train'],20)
    train = df.iloc[subset_train]
    
    print('test Distribution...')
    print(test.label.value_counts())
    
    model, tokenizer, opt = f.init(num_labels=num_labels, model=model_name)
    model.to(device)


    best_acc = 0
    val_acc_old = []
    
    # iter on n epochs
    for j in range(n_epochs):
        #train
        hist_loss = f.batch_pass(
            model=model,
            tokenizer=tokenizer,
            df=train,
            opt=opt,
            criterion=criterion,
            batch_size=batch_size,
            num_labels=num_labels,
            update=True
        )

        #val
        y_pred, y_to_compare, _, _ = f.batch_pass(
            model=model,
            tokenizer=tokenizer,
            df=val,
            batch_size=batch_size,
            num_labels=num_labels,
        )
        
        val_acc = sum(np.asarray(y_pred) == np.asarray(y_to_compare))/len(y_pred)
        val_acc_old = np.append(val_acc_old, val_acc)
        
        if val_acc >= best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'epoch: {j+1:>3}/{n_epochs:>3} loss: {np.round(np.mean(hist_loss), decimals=8):>10} | val: {val_acc:.3%}')
        
    print(f'Best val Acc: {best_acc}')
    return best_acc
import sys
if __name__ == "__main__":
    eval(sys.argv[1])    
    
    



