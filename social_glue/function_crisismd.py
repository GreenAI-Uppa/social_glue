# useful function to train test pytorch model
import os

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from transformers import (BloomForSequenceClassification,
                          BloomTokenizerFast,
                          CamembertForSequenceClassification,
                          CamembertTokenizer,
                          FlaubertForSequenceClassification, FlaubertTokenizer,
                          AutoModelForSequenceClassification, AutoTokenizer)

import config_main as config
import pdb

def inference(model, inputs, labels, opt=None, criterion=None, update=False, hist_loss=None, entropy=False, loss_pred=None):
    """Do inference over the network model

    Args:
        model (_type_): transformers model from hugging face
        inputs (_type_): encoded input thanks to tokenizer
        labels (_type_): label list encoded into integer
        opt (_type_, optional): torch optimizer to update weights. Defaults to None.
        criterion (_type_, optional): torch loss. Defaults to None.
        update (bool, optional): if true it will update weights. Defaults to False.
        hist_loss (list, optional): var to store loss history. Defaults to None.
        entropy (bool, optional): return entropy calculation or not (for no update use case). Defaults to False.
        loss_pred (list, optional): var to store entropy. Defaults to None.

    Returns:
        list: predictions
    """
    preds = None
    if update:
        opt.zero_grad()

    grad = torch.set_grad_enabled(True) if update else torch.no_grad()
    with grad:
        output = model(**inputs, labels=labels) if update else model(**inputs)
        
        if update:
            loss = criterion(output.logits, labels)
            hist_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        
        if entropy:
            proba = torch.softmax(output.logits, dim=1)
            loss_pred += Categorical(probs = proba).entropy().tolist()
        # breaking_ties += np.diff(np.asarray(proba.tolist()), axis=1).flatten().tolist()

    
    if not update:
        _, preds = torch.max(output.logits, 1)
        embed = torch.stack(output.hidden_states[-4:]).sum(0)
        
    return (preds, [np.asarray(e.mean(axis=0).tolist()) for e in embed]) if not update else (None, None)# return sentence embedding 


def batch_pass(model, tokenizer, df, batch_size, num_labels, entropy=False, opt=None, criterion=None, update=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Iterate over df data regarding batch size and n_iter

    Args:
        model (_type_): transformers model from huggingface
        tokenizer (_type_): transformers tokenizer from huggingface
        df (pd.DataFrame): pandas dataframe with a "text" column
        batch_size (int): batch size
        num_labels (int): label number for output classifier
        entropy (bool, optional): return or not entropy calculation. Defaults to False.
        opt (_type_, optional): torch optimizer to train / finetunne model. Defaults to None.
        criterion (_type_, optional): torch loss. Defaults to None.
        update (bool, optional): if true -> will update model's weights. Defaults to False.
        device (str, optional): cuda or cpu training. Defaults to 'cuda'iftorch.cuda.is_available()else'cpu'.

    Returns:
        list: if update will return loss history. if no weight update -> return list with y_pred, y_to_compare, loss_pred (entropy)
    """
    
    n_iter = len(df)//batch_size + 2 #+2 to avoid 0, 1 result and iterate over the last partition
    i_0 = 0
    y_to_compare = []
    y_pred = []
    hist_loss = []
    loss_pred = []
    embs = []
    
    if update:
        model.train()
    else:
        model.eval()

    for i in range(1,n_iter):
        if len(df.text.tolist()[i_0:i*batch_size]) == 0:
            break
        inputs = tokenizer(df.text.tolist()[i_0:i*batch_size], return_tensors="pt", padding="max_length", max_length=80, truncation=True)
        inputs = inputs.to(device)
        #import pdb; pdb.set_trace()
        labels = torch.nn.functional.one_hot(torch.tensor(df.label.values[i_0:i*batch_size]), num_classes=num_labels).to(torch.float)
        labels = labels.to(device)

        preds, emb = inference(model=model, opt=opt, criterion=criterion, inputs=inputs, labels=labels, update=update, hist_loss=hist_loss, entropy=entropy, loss_pred=loss_pred)

        if not update:
            y_to_compare += df.label.values[i_0:i*batch_size].tolist()
            y_pred += preds.tolist()
            embs += emb

        i_0 = i*batch_size

    return hist_loss if update else [y_pred, y_to_compare, loss_pred, embs]

def get_data(path, endswith='.csv', sep=',', **kwargs):
    '''get tweet data'''
    files = list(filter(lambda x: x.endswith(endswith), os.listdir(path)))
    results = []
    df = pd.DataFrame()
    for file in files:
        temp = pd.read_csv(f'{path}/{file}', sep=sep, **kwargs)
        # df = pd.concat([df, temp[["tweetID","crisisname","period","relatedness_annotation","urgency_annotation","intention_annotation"]]], ignore_index=True)
        df = pd.concat([df, temp], ignore_index=True)
    return df        
    
def filter_df(df):
    return df[~df.tweet_text.str.startswith('RT @')]

    
def build_df(df, id_time):
    '''keep only useful variables & add time'''
    flat_id_time = []
    for group in list(filter(lambda x: x is not None, id_time)):
        flat_id_time += list(map(lambda x: [int(x[0]), x[1], x[2]], group)) 

    time_df = pd.DataFrame(flat_id_time, columns=['tweetID', 'created_at', 'text'])
    df = df.join(time_df.set_index('tweetID'), on='tweetID', how='inner')
    df.created_at = pd.to_datetime(df.created_at)

    #filter on time 2007 -> wrong tweet id 
    # df = df[df.created_at.apply(lambda x: x.year!=2007)]
    print(f'dataset size: {len(df):,} lines')
    df = df.sort_values('created_at')

    # df['label'] = np.where((df.class_label == "not_informative") | (df.class_label == "not_humanitarian"), 0, 1)

    print('Global Distribution...')
    print(df.relatedness_annotation.value_counts())
    return df

def init(num_labels, model='bloom'):
    '''init bloom model'''
    print("LOADING MODEL",model)
    if model=='bloom':
        tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m", cache_dir=config.cache_dir)
        model = BloomForSequenceClassification.from_pretrained("bigscience/bloom-560m", num_labels=num_labels, problem_type="multi_label_classification", output_hidden_states=True, cache_dir=config.cache_dir)
    elif model=='flaubert':
        tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased", cache_dir=config.cache_dir)
        model = FlaubertForSequenceClassification.from_pretrained("flaubert/flaubert_base_cased", num_labels=num_labels, output_hidden_states=True, cache_dir=config.cache_dir)
    elif model == 'camembert':
        tokenizer = CamembertTokenizer.from_pretrained("camembert-base", cache_dir=config.cache_dir)
        model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels, output_hidden_states=True, cache_dir=config.cache_dir)
    elif model == 'cam_ft_twitter':
        #model_namet = "/data/pgay/hugging_face_my/camembert-largefinetuned_twitter/"
        model_name = "/data/pgay/hugging_face_my/camembert-largefinetuned_twitter_fake/"
        #model_name = "/data/pgay/temp/"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, output_hidden_states=True)
        #for param in model.roberta.parameters():
        #    param.requires_grad = False
    elif os.path.isdir(model):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=num_labels, output_hidden_states=True)
    else:
        raise ValueError(f"wrong model: {model}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        #lr=1e-5,
        lr=1e-5,
        eps=1e-8,
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=2e-5,
    #     momentum=0.9
    # )

    return model, tokenizer, optimizer


def uniform_split(df, train_size, y_class):
    '''same as train_test_split from sickit-learn but with uniform split. Can't be more than 100% of the smallest class
    pandas dataframe needed
    n: effectif en absolu'''
    if type(y_class) != pd.Series:
        y_class = pd.Series(y_class)
    train = pd.DataFrame()
    
    for y in y_class.unique():
        train = pd.concat([df[y_class == y].sample(n=train_size), train])
    
    return train, df.drop(train.index)
    

def calc_f1_score(df):
    '''add tp/tn column and calc f1 score on binary label'''
    df['contingency'] = np.where(
        (df.pred == 1) & (df.label == 1), 'tp',
        np.where((df.pred != 1) & (df.label != 1), 'tn',
        np.where((df.pred == 1) & (df.label != 1), 'fp',
        np.where((df.pred != 1) & (df.label == 1), 'fn', None)))
    )
    
    contingency_tab = pd.crosstab(df.label,df.pred) #ligne, col
    print(contingency_tab)
    try:
        tp = contingency_tab[1][1] # [pred][true]
        tn = contingency_tab[0][0] # [pred][true]
        fp = contingency_tab[1][0] # [pred][true]
        fn = contingency_tab[0][1] # [pred][true]
    except KeyError:
        return
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(recall+precision)
    
    print(f'''
f1_score  : {f1:.2f}
precision : {precision:.2f}
recall    : {recall:.2f}''')
    
    return df

def show_text(df_test):
    '''print
*********************************
| id | y | y_pred | loss | cont|
|------------------------------|
|                   tweet text |
*********************************'''
    print()
    for ind, _, text, _, y, y_pred, loss, _, cont in df_test.values:
        print(f'''
------------------------------------------------
id                   | y | y_pred | loss | cont
------------------------------------------------
{ind:<20} | {y} | {y_pred:>6} | {loss:>4.3f} | {cont}
------------------------------------------------
{text}
------------------------------------------------
            ''')
