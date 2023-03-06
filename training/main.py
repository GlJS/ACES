import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
from utils import get_data, compute_metrics, tokenize_and_align_labels
from settings import all_labels, data_path, folder

import wandb

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaTokenizerFast, DebertaTokenizerFast
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

import pandas as pd
import torch

MODEL_CHECKPOINTS = ["roberta-large"]
data = pd.read_json(data_path, lines=True)
tags, texts = get_data(data)

id2label = {i: label for i, label in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}

for checkpoint in MODEL_CHECKPOINTS:
    print(f"Training {checkpoint}")
    torch.cuda.empty_cache()


    if "deberta-base" in checkpoint or "deberta-large" in checkpoint:
        tokenizer = DebertaTokenizerFast.from_pretrained(checkpoint, add_prefix_space=True)
    elif "roberta" in checkpoint:
        tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


    dataset = [tokenize_and_align_labels(tokenizer, tokens, labels) for tokens, labels in zip(texts, tags)]
    train, test = train_test_split(dataset, test_size=0.1, random_state=42)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)




    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=f"./output/{checkpoint}",
        evaluation_strategy="steps",
        save_strategy="no",
        learning_rate=1e-5,
        num_train_epochs=5,
        logging_steps=20,
        eval_steps=20,
        auto_find_batch_size=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    wandb.finish()

    del model
    del trainer
    del tokenizer
    del data_collator

