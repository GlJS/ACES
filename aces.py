
import string
import numpy as np


from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union, Dict
from tqdm import tqdm
from transformers import pipeline, TokenClassificationPipeline
from transformers.pipelines.token_classification import AggregationStrategy
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from fense.evaluator import Evaluator
from typing import Literal
from torch.nn import functional as F
import warnings


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_sb = SentenceTransformer('paraphrase-TinyBERT-L6-v2', device=device)
model_sb.eval()

fense_eval = Evaluator(device=device, sbert_model=None)


class ACES(TokenClassificationPipeline):
    def preprocess(self, sentence: str, offset_mapping=None):
        """
        Preprocesses the input sentence by lowercasing, removing punctuation, 
        and preparing the input for the model.
        
        Args:
            sentence (str): The input sentence to process.
            offset_mapping (List[tuple], optional): Offset mapping for token positions.

        Yields:
            Dict: The model inputs prepared for token classification.
        """

        # Lowercase and remove punctuation from the sentence
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()

        # Tokenize the sentence and get model inputs
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=False,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )
        
        # Create an is_last tensor to identify the last batch item
        is_last = torch.zeros((model_inputs["input_ids"].shape[0], 1), dtype=torch.int)
        is_last[-1] = 1
        model_inputs["is_last"] = is_last

        # Add offset mapping if provided
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        # Add the original sentence to the model inputs
        model_inputs["sentence"] = sentence

        yield model_inputs


    def _forward(self, model_inputs: Dict) -> Dict:
        """
        Performs the forward pass of the model.

        Args:
            model_inputs (Dict): The preprocessed model inputs.

        Returns:
            Dict: The outputs from the model including logits, special tokens mask, etc.
        """
        # Pop unnecessary items from model inputs
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")

        # Perform the forward pass of the model
        output = self.model(**model_inputs, output_hidden_states=True, return_dict=True)
        logits = output.logits
        last_hidden_state = output.hidden_states[-1]

        # Return the processed outputs
        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "last_hidden_state": last_hidden_state,
            "is_last": is_last,
            **model_inputs,
        }

    def postprocess(self, model_outputs: Dict, 
                    aggregation_strategy: AggregationStrategy = AggregationStrategy.NONE, 
                    ignore_labels: List[str] = ["O"]):        
        """
        Postprocesses the model outputs to get the final entities.

        Args:
            model_outputs (Dict): The outputs from the model.
            aggregation_strategy (AggregationStrategy, optional): Strategy to aggregate entities.
            ignore_labels (List[str], optional): Labels to ignore during postprocessing.

        Returns:
            Dict: The final entities after postprocessing.
        """
        # Process the model outputs
        if isinstance(model_outputs, list):
            model_outputs = model_outputs[0]
        logits = model_outputs["logits"][0].numpy()

        # Calculate scores from logits
        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        # Filter hiden state
        last_hidden_state = model_outputs["last_hidden_state"][0, model_outputs["special_tokens_mask"][0]!=1]

        # Gather pre-entities
        pre_entities = self.gather_pre_entities(
            model_outputs["sentence"],
            model_outputs["input_ids"][0],
            scores,
            model_outputs["offset_mapping"][0],
            model_outputs["special_tokens_mask"][0].numpy(),
            aggregation_strategy
        )

        assert len(pre_entities) == len(last_hidden_state), "Number of entities and hidden states do not match"

        grouped_entities = self.aggregate(pre_entities)
        # Filter anything that is in self.ignore_labels
        entities = {
            entity: value
            for entity, value in grouped_entities.items()
            if entity not in ignore_labels
        }
        return entities

    def aggregate(self, pre_entities: List[dict]) -> List[dict]:
        """
        Aggregates pre-entities.

        Args:
            pre_entities (List[Dict]): The entities before aggregation.

        Returns:
            List[Dict]: The aggregated entities.
        """

        entities = self.aggregate_words(pre_entities, AggregationStrategy.MAX)
        return self.group_entities_per_pair(entities)


    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        """
        Aggregates entities based on the specified aggregation strategy.

        Args:
            entities (List[Dict]): List of entities to aggregate.
            aggregation_strategy (AggregationStrategy): The strategy to use for aggregation.

        Returns:
            Dict: The aggregated entities based on the specified strategy.
        """
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])

        max_entity = max(entities, key=lambda entity: entity["scores"].max())
        scores = max_entity["scores"]
        idx = scores.argmax()
        score = scores[idx]
        entity = self.model.config.id2label[idx]

        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together adjacent tokens with the same entity prediction.

        Args:
            entities (List[Dict]): The entities predicted by the pipeline.

        Returns:
            Dict: Grouped entities based on similarity.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]

         # Calculate the mean score of the entities
        scores = np.nanmean([entity["score"] for entity in entities])

        # Collect all the tokens
        tokens = [entity["word"] for entity in entities]

        # Create a group for the entities
        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities_per_pair(self, entities: List[dict]) -> List[dict]:
        """
        Groups entities that have the same prediction into pairs.

        Args:
            entities (List[Dict]): The entities predicted by the pipeline.

        Returns:
            List[Dict]: Grouped entities based on the same prediction.
        """
        # Initialize the grouped entities dictionary
        grouped_entities = {entities[0]["entity"]: [[entities[0]]]}
        
        # Iterate through entities to group them
        for i in range(1, len(entities)):
            current_entity = entities[i]["entity"]
            previous_entity = entities[i-1]["entity"]
            
            # Group adjacent entities with the same prediction
            if current_entity == previous_entity:
                grouped_entities[current_entity][-1].append(entities[i])
            else:
                if current_entity not in grouped_entities:
                    grouped_entities[current_entity] = []
                grouped_entities[current_entity].append([entities[i]])
        
        # Apply sub-entity grouping on each entity group
        grouped_entities = {entity: [self.group_sub_entities(sub_entity) for sub_entity in grouped_entities[entity]] 
                            for entity in grouped_entities}

        return grouped_entities


class CalculateACESScore():
    def __init__(self):
        pass

    def combine_references(self, refs: List[dict]) -> dict:
        """
        Combines multiple reference dictionaries into a single dictionary.

        Args:
            refs (List[Dict]): A list of dictionaries containing reference data.

        Returns:
            Dict: A single dictionary with combined values from all dictionaries in the list.
        """
        # Initialize a defaultdict to store combined values
        combined_dicts = defaultdict(list)

        # Iterate over the list of dictionaries in refs[0]
        for d in refs:
            # Iterate over the keys in each dictionary
            for key, value in d.items():
                # If the value is a list, extend the list in the defaultdict
                if isinstance(value, list):
                    combined_dicts[key].extend(value)

        # Convert the defaultdict to a regular dictionary
        result_dict = dict(combined_dicts)

        return result_dict

    def __call__(self, cands: List[dict], refs: Union[List[dict], List[List[dict]]] ) -> List[float]:    
        """
        Calculates ACES scores for each reference against candidates.

        Args:
            cands (List[Dict]): A list of candidate dictionaries.
            refs (Union[List[Dict], List[List[Dict]]]): A list of reference dictionaries or a list of lists of reference dictionaries.

        Returns:
            List[float]: A list of ACES scores for each candidate.
        """
        if isinstance(refs[0], list):
            refs = [self.combine_references(ref) for ref in refs]
        return [self.calculate_aces_score_for_single(cand, ref) for (cand, ref) in zip(cands, refs)]        
    
    def calculate_aces_score_for_single(self, cand: List[dict], ref: List[dict]) -> Tuple[float]:
        """
            Calculates the ACES score for a single candidate and reference.

            Args:
                cand (List[Dict]): A list of candidate dictionaries.
                ref (List[Dict]): A list of reference dictionaries.
            
            Returns:
                Tuple[float]: A tuple containing the ACES score and the number of entities in the candidate.
        """
        scores = []
        # Only calculate the score for the candidate if there are entities in the reference
        overlap = [c for c in cand.keys() if c in [r for r in ref.keys()]]

        # Calculate the score for each entity
        # Batches multiple entities together and calculates the sentence-bert score for each batch.
        if len(overlap) != 0:
            cand_entities = [[r["word"].strip() for r in cand[entity]] for entity in overlap]
            ref_entities = [[c["word"].strip() for c in ref[entity]] for entity in overlap]
            cand_lengths = [len(c) for c in cand_entities]
            ref_lengths = [len(r) for r in ref_entities]
            cand_entities_flattened = [item for sublist in cand_entities for item in sublist]
            ref_entities_flattened = [item for sublist in ref_entities for item in sublist]
            entities = cand_entities_flattened + ref_entities_flattened
            entities_embs = model_sb.encode(entities, convert_to_tensor=True, normalize_embeddings=True)
            cand_entities_embs = entities_embs[:sum(cand_lengths)]
            ref_entities_embs = entities_embs[sum(cand_lengths):]

        # Calculate the score for each entity
        for i in range(len(overlap)):
            if i == 0:
                cand_entity_embs = cand_entities_embs[:len(cand_entities[i])]
                ref_entity_embs = ref_entities_embs[:len(ref_entities[i])]
            else:
                cand_entity_embs = cand_entities_embs[sum(cand_lengths[:i]):sum(cand_lengths[:i+1])]
                ref_entity_embs = ref_entities_embs[sum(ref_lengths[:i]):sum(ref_lengths[:i+1])]

            # Calculate the cosine similarity between the candidate and reference entities
            # The embeddings are normalized, so the cosine similarity is equal to the dot product
            score = cand_entity_embs @ ref_entity_embs.t()


            # Calculate the recall and precision for each entity
            recall_max = torch.max(score, dim=0).values
            precision_max = torch.max(score, dim=1).values
            recall = torch.mean(recall_max)
            precision = torch.mean(precision_max)

            # Emphasize recall over precision
            score = (1 + 9 ** 2) * (precision * recall) / (9 ** 2 * precision + recall)

            scores.append(score.item())

        # Always have a score of 0 if there are no entities in the reference
        if len(scores) == 0:
            scores = [0]
        
        scores = np.mean(scores)
        
        # Penalize for a smaller number of entities
        penalty = ((13 - len(overlap)) / 13) / 1850
        scores = scores - penalty
        scores = scores.item()

        return scores
    
def get_aces_score(cands: List[str], 
                   refs: Union[List[List[str]], List[str]], 
                   model: Literal["base", "large"] = "large",
                   batch_size = 64,
                   average: bool = True,
                   device: int = 0) -> float:
    if model == "base":
        model_name = "gijs/aces-roberta-base-13"
    elif model == "large":
        model_name = "gijs/aces-roberta-13"
    else:
        raise ValueError("Model must be either 'base' or 'large'")
    
    device = device if torch.cuda.is_available() else -1
    pipe = pipeline("token-classification", 
                    model=model_name, 
                    aggregation_strategy="average", 
                    pipeline_class=ACES, 
                    torch_dtype=torch.bfloat16, 
                    device=device)
    if not isinstance(cands, list):
        try:
            cands = cands.tolist()
        except:
            raise ValueError("Candidates must be a list of strings")
    if not isinstance(refs, list):
        try:
            refs = refs.tolist()
        except:
            raise ValueError("References must be a list of strings or a list of lists of strings")

    cands_cas = pipe(cands, batch_size=batch_size)
    

    if isinstance(refs[0], list):
        lengths = list(set([len(r) for r in refs]))
        assert len(lengths) == 1, "All references must have the same number of sentences"
        refs_cas = [r for ref in refs for r in ref]
        refs_cas = pipe(refs_cas, batch_size=batch_size)
        refs_cas = [refs_cas[i:i+lengths[0]] for i in range(0, len(refs_cas), lengths[0])]
    else:
        refs_cas = pipe(refs, batch_size=batch_size)


    aces = CalculateACESScore()
    
    scores = aces(cands_cas, refs_cas)
    scores = np.array(scores)
    print(len(cands))
    fl_err = fense_eval.detect_error_sents(cands, batch_size=batch_size)
    print("Scores:", scores.shape, "Fl_err", fl_err.shape)
    scores = np.array([aces_fl-0.5*err*aces_fl for aces_fl,err in zip(scores, fl_err)])

    if average:
        scores = np.mean(scores)

    return scores


if __name__ == "__main__":
    cands = ["chirping singing and birds a bunch"]
    refs = ["birds are chirping and singing loudly in the forest"]
    scores = get_aces_score(cands, refs, average=False, model="base")
    print(scores)