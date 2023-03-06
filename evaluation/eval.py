import os
import sys
sys.path.insert(0, os.getcwd()) 


from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import numpy as np
import string

from bert_score import score
from aces import ACES, get_aces_score
from evaluation.eval_metrics import evaluate_metrics_from_lists
from transformers import pipeline


# Class to remove all the spam output I get from captioning evaluation tools
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# 2021 dataset is in pickle format. 
# This dataset is the output from the baseline model from the DCASE 2021 challenge.
# Accessible here: https://github.com/audio-captioning/dcase-2021-baseline

with open("./evaluation/captioning-results/captions_pred.pkl", 'rb') as pickle_file:
    captions_pred = pickle.load(pickle_file)
baseline2021 = pd.DataFrame().from_dict(captions_pred)
baseline2021["file_name"] = baseline2021["file_name"].apply(lambda x: x.replace("clotho_file_", "") + ".wav")


# 2022 dataset is in csv format.
passt_evaluation_preds = pd.read_csv("./evaluation/captioning-results/passt-evaluation-preds.csv", sep="\t")
panns_evaluation_preds = pd.read_csv("./evaluation/captioning-results/panns-evaluation-preds.csv", sep="\t")
baseline2022 = pd.read_csv("./evaluation/captioning-results/2022-baseline-evaluation-preds.csv", sep="\t")


# The Clotho dataset to evaluate against
clotho_eval = pd.read_csv("./data/clotho_captions_evaluation.csv")


pipe = pipeline("token-classification", model="output/roberta-large", aggregation_strategy="average", pipeline_class=ACES, device=0)


results = []

for df in [passt_evaluation_preds, panns_evaluation_preds, baseline2021, baseline2022]:
    df_merged = clotho_eval.merge(df, on="file_name")
    df_merged = df_merged.drop_duplicates()
    cands = df_merged["caption_predicted"].tolist()
    refs = df_merged[["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]].values.tolist()

    # ACES
    aces_output = get_aces_score(cands, refs, average=True, pipe=pipe)
    aces_output = np.nanmean(aces_output)
    aces_df = pd.DataFrame([aces_output], columns=["aces"])


    # Caption Evaluation Metrics
    res_metrics = []
    with HiddenPrints():
        res = evaluate_metrics_from_lists(cands, refs)[0]
        res_metrics.append(res)

    # BERTScore
    p2, r2, f2 = score(cands, refs, lang="en")
    res_out = pd.DataFrame(res_metrics)
    res_out = pd.concat([res_out, aces_df], axis=1)

    res_out["f1_bert"] = f2.mean().item()
    results.append(res_out)

results = pd.concat(results)
indices = [f"{x}" for x in ["passt", "panns", "baseline2021", "baseline2022"]]
results.index = indices
results.to_csv("./evaluation/captioning-results/output/results.csv")