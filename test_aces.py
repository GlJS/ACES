from aces import get_aces_score, ACES
from transformers import pipeline


def test_single_single():
    cands = ["a bunch of birds are singing"]
    refs = ["birds are chirping and singing loudly in the forest"]
    pipe = pipeline("token-classification", model="gijs/aces-roberta-13", aggregation_strategy="average", pipeline_class=ACES, device=-1)
    scores = get_aces_score(cands, refs, average=False, pipe=pipe)
    assert all(isinstance(x, float) for x in scores)
    print(scores)

def test_single():
    cands = ["a bunch of birds are singing"]
    refs = ["birds are chirping loudly in the forest while insects are buzzing", "many birds are chirping in the distance while other animals also make noise", "many birds chirping in the distance with other animals making noises", "many different birds are chirping and tweeting together", "several different birds are chirping harmoniously in nature"]
    pipe = pipeline("token-classification", model="gijs/aces-roberta-13", aggregation_strategy="average", pipeline_class=ACES, device=-1)
    scores = get_aces_score(cands, refs, average=False, pipe=pipe)
    assert all(isinstance(x, float) for x in scores)
    print(scores)

def test_multiple_single():
    cands = ["a bunch of birds are singing", "multiple cars pass by on a nearby street"]
    refs = [["birds are chirping loudly in the forest while insects are buzzing"], ["cars and trucks passing by on a nearby street"]]
    pipe = pipeline("token-classification", model="gijs/aces-roberta-13", aggregation_strategy="average", pipeline_class=ACES, device=-1)
    scores = get_aces_score(cands, refs, average=False, pipe=pipe)
    assert all(isinstance(x, float) for x in scores)
    print(scores)

def test_multiple_single():
    cands = ["a bunch of birds are singing", "multiple cars pass by on a nearby street"]
    refs = [["birds are chirping loudly in the forest while insects are buzzing"], ["cars and trucks passing by on a nearby street"]]
    pipe = pipeline("token-classification", model="gijs/aces-roberta-13", aggregation_strategy="average", pipeline_class=ACES, device=-1)
    scores = get_aces_score(cands, refs, average=False, pipe=pipe)
    assert all(isinstance(x, float) for x in scores)
    print(scores)
        
def test_multiple_multiple():
    cands = ["a bunch of birds are singing", "multiple cars pass by on a nearby street"]
    refs = [["birds are chirping loudly in the forest while insects are buzzing", "many birds are chirping in the distance while other animals also make noise", "many birds chirping in the distance with other animals making noises", "many different birds are chirping and tweeting together", "several different birds are chirping harmoniously in nature"], ["a vehicle passing by smoothly then stops abruptly and starts accelerating again", "a whining vehicle interrupts a clattering object, while a strong wind blows", "cars and trucks passing by on a nearby street", "vehicles buzz while travelling along a busy street during a windy day", "vehicles on a nearby street are passing by"]]
    pipe = pipeline("token-classification", model="gijs/aces-roberta-13", aggregation_strategy="average", pipeline_class=ACES, device=-1)
    scores = get_aces_score(cands, refs, average=False, pipe=pipe)
    assert all(isinstance(x, float) for x in scores)
    print(scores)


