import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
from utils import get_data, compute_metrics, tokenize_and_align_labels
from settings import all_labels, data_path, reduced
import wandb

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaTokenizerFast, DebertaTokenizerFast
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer

import pandas as pd
import torch

MODEL_CHECKPOINTS = ["roberta-base"]
data = pd.read_json(data_path, lines=True)
tags, texts = get_data(data)

id2label = {i: label for i, label in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}

for checkpoint in MODEL_CHECKPOINTS:
    print(f"Training {checkpoint}")
    torch.cuda.empty_cache()


    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint, add_prefix_space=True)


    dataset = [tokenize_and_align_labels(tokenizer, tokens, labels) for tokens, labels in zip(texts, tags)]
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)




    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=f"./output/hub/{checkpoint}{'_reduced' if reduced else ''}",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        num_train_epochs=5,
        logging_steps=10,
        weight_decay=0.01,
        report_to="wandb",
        push_to_hub=True,
        push_to_hub_model_id=f"roberta-base{'-reduced' if reduced else ''}",
        run_name=f"{checkpoint}{'_reduced' if reduced else ''}",
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