import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,1"
import sys
sys.path.insert(0, os.getcwd()) 


from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import numpy as np
import string

from bert_score import score
from src.aces.aces import ACES, get_aces_score
from evaluation.eval_metrics import evaluate_metrics_from_lists
from transformers import pipeline
from fense.evaluator import Evaluator


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
baseline2023 = pd.read_csv("./evaluation/captioning-results/2023-baseline-evaluation-preds.csv", sep="\t")

audiocaption = pd.read_csv("./evaluation/captioning-results/audiocaption-preds.csv")


# The Clotho dataset to evaluate against
clotho_eval = pd.read_csv("./dataset/clotho_captions_evaluation.csv")


pipe = pipeline("token-classification", model="gijs/aces-roberta-13", aggregation_strategy="simple", pipeline_class=ACES, device=0)
pipe2 = pipeline("token-classification", model="gijs/aces-roberta-base-13", aggregation_strategy="simple", pipeline_class=ACES, device=1)


results = []

all_csvs = ["baseline2021", "audiocaption", "baseline2022", "panns", "passt", "baseline2023"]

for df in [baseline2021, audiocaption, baseline2022, panns_evaluation_preds, passt_evaluation_preds, baseline2023]:
    df_merged = clotho_eval.merge(df, on="file_name")
    df_merged = df_merged.drop_duplicates()
    cands = df_merged["caption_predicted"].tolist()
    refs = df_merged[["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]].values.tolist()

    # ACES
    aces_output = get_aces_score(cands, refs, average=True)
    aces_df = pd.DataFrame([aces_output], columns=["aces"])

    aces_output2 = get_aces_score(cands, refs, average=True)
    aces_df2 = pd.DataFrame([aces_output2], columns=["aces2"])

    # FENSE
    fense_eval = Evaluator(device='cpu', sbert_model='paraphrase-MiniLM-L6-v2', echecker_model='echecker_clotho_audiocaps_tiny')
    scores = []
    for cand, ref in zip(cands, refs):
        _, _, penalized_score = fense_eval.sentence_score(cand, ref, return_error_prob=True)
        scores.append(penalized_score)
    fense_df = pd.DataFrame([np.mean(scores)], columns=["fense"])



    # Caption Evaluation Metrics
    res_metrics = []
    with HiddenPrints():
        res = evaluate_metrics_from_lists(cands, refs)[0]
        res_metrics.append(res)

    # BERTScore
    p2, r2, f2 = score(cands, refs, lang="en")
    res_out = pd.DataFrame(res_metrics)
    res_out = pd.concat([res_out, aces_df, fense_df], axis=1)

    res_out["f1_bert"] = f2.mean().item()
    results.append(res_out)

results = pd.concat(results)
indices = [f"{x}" for x in all_csvs]
results.index = indices
results.to_csv("evaluation/captioning-results/output/results_acesv2.csv")