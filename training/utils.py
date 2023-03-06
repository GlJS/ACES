import numpy as np
import pandas as pd
from itertools import combinations
from settings import all_labels
import evaluate
metric = evaluate.load("poseval")


def get_data(data):
    
    data = data.drop_duplicates(subset=['text'])
    omit_label = all_labels.index("O")
    tags = [[omit_label for token in tokens] for tokens in data["tokens"]]
    texts = [[t["text"] for t in text] for text in data["tokens"]]

    for idx, spans in enumerate(data["spans"]):
        for span in spans:
            for jdx in range(span["token_start"], span["token_end"]+1):
                tags[idx][jdx] = all_labels.index(span["label"])

    return tags, texts


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[all_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    # all_metric_keys = [i for i all_metrics if not i in ["accuracy", "weighted avg", "macro_avg"]
    return {
        "precision": all_metrics["weighted avg"]["precision"],
        "recall": all_metrics["weighted avg"]["recall"],
        "f1": all_metrics["weighted avg"]["f1-score"],
        "accuracy": all_metrics["accuracy"],
        "f1_who": all_metrics["WHO"]["f1-score"],
        "f1_what": all_metrics["WHAT"]["f1-score"],
        "f1_where": all_metrics["WHERE"]["f1-score"],
        "f1_how": all_metrics["HOW"]["f1-score"],
    }

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            new_labels.append(label)

    return new_labels



def tokenize_and_align_labels(tokenizer, tokens, labels):
    tokenized_inputs = tokenizer(
        tokens, truncation=True, is_split_into_words=True
    )
    word_ids = tokenized_inputs.word_ids()

    tokenized_inputs["labels"] = align_labels_with_tokens(labels, word_ids)
    return tokenized_inputs

