# ACES

This is the repository of Audio Captioning Evaluation on Semantics of Sound (ACES). 

In here you will find the instructions how to train an ACES model and calculate statistics. 

## Installation
```
pip install aces-metric
```
## Usage
The candidates can be a list, the references can be a list or a list of lists. 
```
from aces import get_aces_score
candidates = ["a bunch of birds are singing"]
references = ["birds are chirping and singing loudly in the forest"]
score = get_aces_score(candidates, references, average=True)
```

## Evaluation
All the code that is used to evaluate different models for the research paper can be found in the `evaluation` folder. Particularly,
the model evaluation can be found in `evaluation/eval.py`, and information about the FENSE experiment can be found in `evaluation/fense_experiment/main.py`. 
