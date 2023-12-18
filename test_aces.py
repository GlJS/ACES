import os
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from src.aces.aces import get_aces_score, ACES
from transformers import pipeline
import time
from evaluation.fense_experiment.experiment.sweep import get_aces_score as get_aces_score_old


def test_single_single():
    cands = ["a bunch of birds are singing"]
    refs = ["birds are chirping and singing loudly in the forest"]
    scores = get_aces_score(cands, refs, average=False)
    assert all(isinstance(x, float) for x in scores)

    print("scores.shape", scores.shape)

def test_single():
    cands = ["a bunch of birds are singing"]
    refs = ["birds are chirping loudly in the forest while insects are buzzing", "many birds are chirping in the distance while other animals also make noise", "many birds chirping in the distance with other animals making noises", "many different birds are chirping and tweeting together", "several different birds are chirping harmoniously in nature"]
    scores = get_aces_score(cands, refs, average=False)

    assert all(isinstance(x, float) for x in scores)
    print("scores.shape", scores.shape)

def test_multiple_single():
    cands = ["a bunch of birds are singing", "multiple cars pass by on a nearby street"]
    refs = [["birds are chirping loudly in the forest while insects are buzzing"], ["cars and trucks passing by on a nearby street"]]
    scores = get_aces_score(cands, refs, average=False)

    assert all(isinstance(x, float) for x in scores)
    print("scores.shape", scores.shape)

def test_multiple_single():
    cands = ["a bunch of birds are singing", "multiple cars pass by on a nearby street"]
    refs = [["birds are chirping loudly in the forest while insects are buzzing"], ["cars and trucks passing by on a nearby street"]]
    scores = get_aces_score(cands, refs, average=False)

    assert all(isinstance(x, float) for x in scores)
    print("scores.shape", scores.shape)
        
def test_multiple_multiple():
    cands = ["a bunch of birds are singing", "multiple cars pass by on a nearby street"]
    refs = [["birds are chirping loudly in the forest while insects are buzzing", "many birds are chirping in the distance while other animals also make noise", "many birds chirping in the distance with other animals making noises", "many different birds are chirping and tweeting together", "several different birds are chirping harmoniously in nature"], ["a vehicle passing by smoothly then stops abruptly and starts accelerating again", "a whining vehicle interrupts a clattering object, while a strong wind blows", "cars and trucks passing by on a nearby street", "vehicles buzz while travelling along a busy street during a windy day", "vehicles on a nearby street are passing by"]]
    scores = get_aces_score(cands, refs, average=False)

    assert all(isinstance(x, float) for x in scores)
    print("scores.shape", scores.shape)

def test_clotho():
    import pandas as pd
    clotho_eval = pd.read_csv("./dataset/clotho_captions_evaluation.csv")
    cands = clotho_eval["caption_1"].tolist()
    refs = clotho_eval[["caption_2", "caption_3", "caption_4", "caption_5"]].values.tolist()
    scores = get_aces_score(cands, refs, average=False)

    assert all(isinstance(x, float) for x in scores)
    print("scores.shape", scores.shape)


def test_clotho_single():
    import pandas as pd
    clotho_eval = pd.read_csv("./dataset/clotho_captions_evaluation.csv")
    cands = clotho_eval["caption_1"].tolist()
    refs = clotho_eval["caption_2"].tolist()
    scores = get_aces_score(cands, refs, average=True)
    assert isinstance(scores, float)
    print("scores_single", scores)

def test_clotho_single_old():
    import pandas as pd
    clotho_eval = pd.read_csv("./dataset/clotho_captions_evaluation.csv")
    cands = clotho_eval["caption_1"].tolist()
    refs = clotho_eval["caption_2"].tolist()
    pipe = pipeline("token-classification", model="gijs/aces-roberta-13", aggregation_strategy="simple", pipeline_class=ACES, device=2)

    scores = get_aces_score_old(cands, refs, average=True, pipe=pipe, model="gijs/aces-roberta-13", division=0.9985719914845788,
                                fl_weighing = True, f1_beta = 9, f1 = 3.797569713186928, average_strategy = "simple", use_sbert = True, 
                                f1_calc = "max-mean", penalty_score = 1850, apply_penalty = True, overlap_type = "both", 
                                f1_weight = 0.5000509621781535, distance_technique = "cosine", use_score = "no", 
                                score_weighing = 0.5, overall_sbert = False, overall_sbert_weight = 0.5, sbert_based_on_scores = True)
    scores = np.mean(scores)
    assert isinstance(scores, float)
    print("scores_single", scores)


def test_clotho_average():
    import pandas as pd
    clotho_eval = pd.read_csv("./dataset/clotho_captions_evaluation.csv")
    cands = clotho_eval["caption_1"].tolist()
    refs = clotho_eval[["caption_2", "caption_3", "caption_4", "caption_5"]].values.tolist()
    scores = get_aces_score(cands, refs, average=True)
    assert isinstance(scores, float)
    print("scores_single", scores)


def test_same():
    cands = ["a bunch of birds are singing"]
    refs = ["a bunch of birds are singing"]
    scores = get_aces_score(cands, refs, average=False)
    assert all(isinstance(x, float) for x in scores)
    print("scores.shape", scores.shape)

if __name__ == "__main__":
    start_time = time.time()
    # test_single_single()
    # test_single()
    # test_multiple_single()
    # test_multiple_multiple()
    # test_clotho()
    test_clotho_single()
    # test_clotho_single_old()
    # test_clotho_average()

    print("--- %s seconds ---" % (time.time() - start_time))