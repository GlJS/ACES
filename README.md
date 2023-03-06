# ACES

This is the repository of Audio Captioning Evaluation on Semantics of Sound (ACES). 

In here you will find the instructions how to train an ACES model and calculate statistics. 

## Installation
```
git clone https://github.com/GlJS/ACES.git
cd ACES
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Usage
First, train a model running `python3 training/train.py`. Then you can import ACES by:
```
from transformers import pipeline
from aces import ACES, get_aces_score
cands = ["Young woman talking with crickling noise"]
refs = ["Paper crackling with female speaking lightly in the background"]
pipe = pipeline("token-classification", model="output/roberta-large", aggregation_strategy="average", pipeline_class=ACES, device=0)
scores = get_aces_score(cands, refs, average=False, pipe=pipe)
print(scores)
```

## Evaluation
```
Model evaluation can be found in `evaluation/eval.py`, and information about the FENSE experiment can be found in `evaluation/fense-experiment/main.py`. 