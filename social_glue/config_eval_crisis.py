import json, os

CRISIS_MD_DIR = '/data/pgay/social_computing/glue_social/crisis_md/' 
split_file = 'crisis_splits.json'
splits = json.load(open(os.path.join(CRISIS_MD_DIR,split_file)))

# evaluation for the classification
num_labels = 3
batch_size_crisismd = 4 
n_epochs = 5
