
import string
import numpy as np


from typing import Any, List, Optional, Tuple, Union
from tqdm import tqdm
from transformers import pipeline, TokenClassificationPipeline
from transformers.pipelines.token_classification import AggregationStrategy
import torch


class ACES(TokenClassificationPipeline):
    def preprocess(self, sentence, offset_mapping=None):
        # lowercase and remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=False,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
        )
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        model_inputs["sentence"] = sentence

        return model_inputs


    def _forward(self, model_inputs):
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")

        output = self.model(**model_inputs, output_hidden_states=True, return_dict=True)
        logits = output.logits
        last_hidden_state = output.hidden_states[-1]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "last_hidden_state": last_hidden_state,
            **model_inputs,
        }

    def postprocess(self, model_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=["O"]):
        logits = model_outputs["logits"][0].numpy()

        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


        last_hidden_state = model_outputs["last_hidden_state"][0, model_outputs["special_tokens_mask"][0]!=1]

        pre_entities = self.gather_pre_entities(
            model_outputs["sentence"],
            model_outputs["input_ids"][0],
            scores,
            model_outputs["offset_mapping"][0],
            model_outputs["special_tokens_mask"][0].numpy(),
            aggregation_strategy
        )

        assert len(pre_entities) == len(last_hidden_state), "Number of entities and hidden states do not match"
        for idx, entity in enumerate(pre_entities):
            entity["hidden_state"] = last_hidden_state[idx]

        grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
        # Filter anything that is in self.ignore_labels
        entities = [
            entity
            for entity in grouped_entities
            if entity.get("entity", None) not in ignore_labels
            and entity.get("entity_group", None) not in ignore_labels
        ]
        return entities


    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.AVERAGE:
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            raise ValueError("Invalid aggregation_strategy")
        new_entity = {
            "entity": entity,
            "hidden_state": torch.stack([entity["hidden_state"] for entity in entities], axis=0),
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        hidden_state = torch.cat([entity["hidden_state"] for entity in entities], axis=0)

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "hidden_state": hidden_state,
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the tokens with the same entity predicted.
        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        grouped_entities = {}
        for entity in entities:
            if entity["entity"] not in grouped_entities:
                grouped_entities[entity["entity"]] = []
            grouped_entities[entity["entity"]].append(entity)

        grouped_entities = [self.group_sub_entities(grouped_entities[entity]) for entity in grouped_entities]

        return grouped_entities

class CalculateACESScore():
    def __init__(self):
        pass

    def __call__(self, cands: List[List[dict]], refs: Union[List[List[dict]], List[List[List[dict]]]] ) -> List[float]:
        if isinstance(refs[0][0], list):
            cands, refs, groups = self.flatten(cands, refs)
        else:
            groups = None
    
        scores = []
        for idx, (cand, ref) in tqdm(enumerate(zip(cands, refs))):    
            score = self.calculate_aces_score_for_single(cand, ref)
            scores.append(score)
        
        if groups is not None:
            scores = self.unflatten(scores, groups)

        return scores

    def flatten(self, cands: List[List[dict]], refs: List[List[List[dict]]]) -> Tuple[List[List[dict]], List[List[dict]], List[int]]:
        """
        Flatten the references and candidates to have the same structure.

        Args:
            cands (`List[List[dict]]`): The candidates.
            refs (`List[List[List[dict]]]`): The references.
        """
        new_cands = []
        new_refs = []
        new_groups = []
        for idx, (cand, ref) in enumerate(zip(cands, refs)):
            for ref_ in ref:
                new_cands.append(cand)
                new_refs.append(ref_)
                new_groups.append(idx)

        return new_cands, new_refs, new_groups
    
    def unflatten(self, scores: List[float], groups: List[int]) -> List[float]:
        new_scores = {}
        for idx, group in enumerate(groups):
            if group not in new_scores:
                new_scores[group] = []
            new_scores[group].append(scores[idx])
        new_scores = np.array(list(new_scores.values()))
        new_scores = np.nanmean(new_scores, axis=1)
        return new_scores
        
    
    def calculate_aces_score_for_single(self, cand: List[dict], ref: List[dict]) -> Tuple[float]:
        scores = []
        for cand_entity in cand:
            correct_ref_entity = list(filter(lambda ref_entity: ref_entity["entity_group"] == cand_entity["entity_group"], ref))
            assert len(correct_ref_entity) == 1, "There should be only one reference entity for each candidate entity"
            ref_entity = correct_ref_entity[0]
            cand_norm = cand_entity["hidden_state"] / cand_entity["hidden_state"].norm(dim=-1, keepdim=True)
            ref_norm = ref_entity["hidden_state"] / ref_entity["hidden_state"].norm(dim=-1, keepdim=True)

            score = torch.mm(cand_norm, ref_norm.transpose(0, 1)).squeeze()
            scores.append(score)
        
        ref_ = [r["entity_group"] for r in ref]
        cand_ = [c["entity_group"] for c in cand]
        try:
            precision = len(set(ref_).intersection(set(cand_))) / len(cand_)
            recall = len(set(ref_).intersection(set(cand_))) / len(ref_)
            f1 = 2 * (precision * recall) / (precision + recall)
        except:
            precision = 0
            recall = 0
            f1 = 0

        scores = [torch.mean(score).item() for score in scores]
        scores = np.mean(scores)

        return scores, f1
    
def get_aces_score(cands: List[str], refs: Union[List[List[str]], List[str]], average=False,
                   pipe=None) -> float:
    if not isinstance(cands, list):
        cands = cands.tolist()
    if not isinstance(refs, list):
        refs = refs.tolist()
    cands_cas = pipe(cands, batch_size=64)
    

    if isinstance(refs[0], list):
        refs_cas = [pipe(ref, batch_size=64) for ref in refs]
    else:
        refs_cas = pipe(refs, batch_size=64)


    aces = CalculateACESScore()
    scores = aces(cands_cas, refs_cas)
    if average:
        scores = np.nanmean(scores, axis=0)
        return (scores[0] + scores[1]) / 2
    else:
        scores = np.array(scores)
        return (scores[:, 0] + scores[:, 1]) / 2


if __name__ == "__main__":
    cands = ["Young woman talking with crickling noise"]
    refs = ["Paper crackling with female speaking lightly in the background"]
    pipe = pipeline("token-classification", model="output/roberta-large", aggregation_strategy="average", pipeline_class=ACES, device=1)
    scores = get_aces_score(cands, refs, average=False, pipe=pipe)
    print(scores)