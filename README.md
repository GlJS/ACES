# ACES

This is the repository of Audio Captioning Evaluation on Semantics of Sound (ACES). 

In here you will find the instructions how to train an ACES model and calculate statistics. 

## Installation
``` bash
pip install aces-metric
```
## Usage
The candidates can be a list, the references can be a list or a list of lists. 
``` python
from aces import get_aces_score
candidates = ["a bunch of birds are singing"]
references = ["birds are chirping and singing loudly in the forest"]
score = get_aces_score(candidates, references, average=True)
```

### Semantics of sounds
To get an output of classes of semantic groups from a caption:
``` python
from transformers import pipeline
pipe = pipeline("token-classification", "gijs/aces-roberta-13", aggregation_strategy="simple")
pipe("Bird chirps in the tree while a car hums")
```

## Evaluation
All the code that is used to evaluate different models for the research paper can be found in the `evaluation` folder on the [github](https://github.com/GlJS/ACES). Particularly, the model evaluation can be found in `evaluation/eval.py`, and information about the FENSE experiment can be found in `evaluation/fense_experiment/main.py`. 
