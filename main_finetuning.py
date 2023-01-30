import config, json
import dataset_finetuning
import eval_crisismd
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
import math
import sys
from transformers import AutoModelForMaskedLM
from accelerate import Accelerator
from transformers import get_scheduler

# language model configuration
model = AutoModelForMaskedLM.from_pretrained(config.model_checkpoint, cache_dir=config.cache_dir)
optimizer = AdamW(model.parameters(), lr=5e-5)
accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, dataset_finetuning.train_dataloader, dataset_finetuning.eval_dataloader
)

# num_training steps for the scheduler
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = config.num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))
results = {}
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(config.output_dir, save_function=accelerator.save)
crisismd_val = eval_crisismd.eval(config.output_dir)
print("EPOCH -1 ACCURACY CRISISMD:",crisismd_val)
results[-1] = {"perplexity":0,"crisismd_val":crisismd_val}

for epoch in range(config.num_train_epochs):
    # Training
    model.train()
    for i, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if i%int(num_training_steps/10) == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(config.output_dir, save_function=accelerator.save)
            crisismd_val = eval_crisismd.eval(config.output_dir)
            print("EPOCH ", epoch,i,"ACCURACY CRISISMD",crisismd_val)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(config.batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(dataset_finetuning.eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(config.output_dir, save_function=accelerator.save)
    #if accelerator.is_main_process:
    #    tokenizer.save_pretrained(output_dir)
    # compute crisismd accuracy
    crisismd_val = eval_crisismd.eval(config.output_dir)
    print("EPOCH ", epoch,"ACCURACY CRISISMD",crisismd_val)
    results[epoch] = {"perplexity":perplexity,"crisismd_val":crisismd_val}


json.dump(results,open("results.json","w"))

#https://huggingface.co/course/chapter7/3?fw=pt
