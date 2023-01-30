import os
import config
from transformers import AutoModelForMaskedLM
from datasets import load_dataset


# pip install protobuf==3.20.*
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

files = ["/data/pgay/temp.txt"]

twitter_dataset = load_dataset("text", data_files=files)

def preprocess(text):
    new_text = []
    for t in text['text'].split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return {'text' : " ".join(new_text)}

twitter_dataset = twitter_dataset.map(preprocess)
sample = twitter_dataset["train"].shuffle(seed=42).select(range(3))
for row in sample:
    print(f"\n'>>> Tweet: {row['text']}'")


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result
# Use batched=True to activate fast multithreading!
tokenized_datasets = twitter_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)


chunk_size = 64

# Drop the last chunk if it’s smaller than chunk_size.
def group_texts(examples):
    """
    So you concatenate all the texts in your dataset to have a very long sentence
    And then, you split this sentence into chunks of equal size
    and you drop the last chunk
    + create a new labels column which is a copy of the input_ids one, that will be the GT for the language modelling task
    """
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)
print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

"""
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
"""


## MASK WHOLE WORLDS
import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


train_size = int(len(lm_datasets['train']) * 0.9)
test_size = int(0.1 * train_size)
print(train_size, test_size)
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)


from transformers import TrainingArguments

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    #masked_inputs = data_collator(features)
    masked_inputs = whole_word_masking_data_collator(features)
    # Create a new "masked" column for each column in the dataset
    #return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)

eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)


from torch.utils.data import DataLoader
from transformers import default_data_collator



train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=config.batch_size,
    collate_fn=whole_word_masking_data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, collate_fn=default_data_collator
)

#https://huggingface.co/course/chapter7/3?fw=pt
