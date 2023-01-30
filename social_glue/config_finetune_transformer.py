# crisismd evaluation
import torch
import json, os
split_file = 'crisis_splits.json'
splits = json.load(open(split_file))


# language model configuration
from transformers import AutoModelForMaskedLM

model_checkpoint = "camembert/camembert-large"
model_name = model_checkpoint.split("/")[-1]+'finetuned_twitter'
output_dir = os.path.join('/mnt/beegfs/home/gay/hugging_face_my/',model_name)
batch_size = 64

model_dir = '/data/pgay/hugging_face_my/'

num_train_epochs = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# evaluation for the classification
num_labels = 3
batch_size_crisismd = 32
n_epochs = 2
