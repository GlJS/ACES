import os
import sys
sys.path.insert(0, os.getcwd()) 
from main import print_accuracy
from dataloader import get_former, get_latter
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from bert_score import BERTScorer
from bleurt import score as bleurt_score
from src.aces.fense.evaluator import Evaluator
from evaluation.eval_metrics import evaluate_metrics_from_lists
import wandb
import string
from transformers import pipeline, TokenClassificationPipeline
from transformers.pipelines.token_classification import AggregationStrategy
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import warnings



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device_num = 0

model_sb = SentenceTransformer('paraphrase-TinyBERT-L6-v2', device=device)
model_sb.eval()

scorer_bs = BERTScorer(model_type='ramsrigouthamg/t5_paraphraser', lang="en", rescale_with_baseline=False, device=device2)

scorer_brt = bleurt_score.BleurtScorer()

fense_eval = Evaluator(device=device2 if torch.cuda.is_available() else 'cpu', sbert_model=None)


class ACES(TokenClassificationPipeline):
    def preprocess(self, sentence, offset_mapping=None):
        # lowercase and remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=False,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast
        )
        # add is_last
        is_last = torch.zeros((model_inputs["input_ids"].shape[0], 1), dtype=torch.int)
        is_last[-1] = 1
        model_inputs["is_last"] = is_last

        
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        model_inputs["sentence"] = sentence

        yield model_inputs


    def _forward(self, model_inputs):
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")

        output = self.model(**model_inputs, output_hidden_states=True, return_dict=True)
        logits = output.logits
        last_hidden_state = output.hidden_states[-1]


        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "last_hidden_state": last_hidden_state,
            "is_last": is_last,
            **model_inputs,
        }

    def postprocess(self, model_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=["O"]):
        if isinstance(model_outputs, list):
            model_outputs = model_outputs[0]
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
        entities = {
            entity: value
            for entity, value in grouped_entities.items()
            if entity not in ignore_labels
        }
        return entities

    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "hidden_state": pre_entity["hidden_state"],
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        return self.group_entities_per_pair(entities)


    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
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

        if entities[0]["hidden_state"].dim() == 1:
            hidden_state = torch.stack([entity["hidden_state"] for entity in entities], axis=0)
        else:
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

    def group_entities_per_pair(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the tokens with the same entity predicted.
        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        grouped_entities = {entities[0]["entity"]: [[entities[0]]]}
        
        for i in range(1, len(entities)):
            if entities[i]["entity"] == entities[i-1]["entity"]:
                grouped_entities[entities[i]["entity"]][-1].append(entities[i])
            else:
                if entities[i]["entity"] not in grouped_entities:
                    grouped_entities[entities[i]["entity"]] = []
                grouped_entities[entities[i]["entity"]].append([entities[i]])
        

        grouped_entities = {entity: [self.group_sub_entities(sub_entity) for sub_entity in grouped_entities[entity]] for entity in grouped_entities}

        return grouped_entities


class CalculateACESScore():
    def __init__(self, **kwargs):
        self.labels = 13 if kwargs['model'].endswith("13") else 10
        self.f1_calc = kwargs['f1_calc']
        self.f1_beta = kwargs['f1_beta']
        self.f1 = kwargs["f1"]
        self.weighing = kwargs['division']
        self.use_sbert = kwargs['use_sbert']
        self.penalty_score = kwargs["penalty_score"]
        self.overlap_type = kwargs["overlap_type"]
        self.distance_technique = kwargs["distance_technique"]
        self.apply_penalty = kwargs["apply_penalty"]
        self.use_score = kwargs["use_score"]
        self.score_weighing = kwargs["score_weighing"]




    def __call__(self, cands: List[List[dict]], refs: Union[List[List[dict]], List[List[List[dict]]]]) -> List[float]:
        scores = []
        # cands, refs, groups = self.flatten(cands, refs)
        for idx, (cand, ref) in tqdm(enumerate(zip(cands, refs))):    
            score = self.calculate_aces_score_for_single(cand, ref)
            scores.append(score)

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
        
    
    def calculate_aces_score_for_single(self, cand, ref) -> Tuple[float]:
        scores = []
        ref_scores = []
        cand_scores = []
        overlap = [c for c in cand.keys() if c in [r for r in ref.keys()]]
        for entity in overlap:
            ref_entity = ref[entity]
            cand_entity = cand[entity]

            ref_entity_txts = [r["word"].strip() for r in ref_entity]
            cand_entity_txts = [c["word"].strip() for c in cand_entity]

            if self.use_sbert:
                ref_entity_embs = model_sb.encode(ref_entity_txts, convert_to_tensor=True)
                cand_entity_embs = model_sb.encode(cand_entity_txts, convert_to_tensor=True)
            else:
                ref_entity_embs = torch.cat([r["hidden_state"] for r in ref_entity], dim=0)
                cand_entity_embs = torch.cat([c["hidden_state"] for c in cand_entity], dim=0)


            if self.distance_technique == "cosine":
                score = F.normalize(cand_entity_embs, dim=-1) @ F.normalize(ref_entity_embs, dim=-1).t()
            elif self.distance_technique == "euclidean":
                score = torch.cdist(cand_entity_embs, ref_entity_embs, p=2)
                score = 1 - (score / torch.max(score))
            else:
                raise ValueError("Invalid distance technique")
            

            ref_entity_scores = [r["score"] for r in ref_entity]
            cand_entity_scores = [c["score"] for c in cand_entity]
            
            ref_scores.append(ref_entity_scores)
            cand_scores.append(cand_entity_scores)

                    
            if self.f1_calc == "mean":
                score = torch.mean(score)
            elif self.f1_calc == "mean-max":
                recall_mean = torch.mean(score, dim=0)
                precision_mean = torch.mean(score, dim=1)
                recall = torch.max(recall_mean)
                precision = torch.max(precision_mean)

                score = (1 + self.f1_beta ** 2) * (precision * recall) / (self.f1_beta ** 2 * precision + recall)
            elif self.f1_calc == "max-mean":
                recall_max = torch.max(score, dim=0).values
                precision_max = torch.max(score, dim=1).values
                recall = torch.mean(recall_max)
                precision = torch.mean(precision_max)
                
                score = (1 + self.f1_beta ** 2) * (precision * recall) / (self.f1_beta ** 2 * precision + recall)
            elif self.f1_calc == "max-max":
                recall_max = torch.max(score, dim=0).values
                precision_max = torch.max(score, dim=1).values
                recall = torch.max(recall_max)
                precision = torch.max(precision_max)

                score = (1 + self.f1_beta ** 2) * (precision * recall) / (self.f1_beta ** 2 * precision + recall)
            else:
                raise ValueError("Invalid f1_calc")
            
            if self.use_score == "pairwise":
                score = score * self.score_weighing + (1 - self.score_weighing) * (torch.mean(torch.Tensor(ref_entity_scores)) + torch.mean(torch.Tensor(cand_entity_scores))) / 2

            
            scores.append(score.item())
        if len(scores) == 0:
            scores = [0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                ref_score = np.array([np.array(r).mean() for r in ref_scores]).mean()
                cand_score = np.array([np.array(c).mean() for c in cand_scores]).mean()
            except:
                ref_score = 0
                cand_score = 0

        scores = np.mean(scores)

        if self.use_score == "mean":
            scores = scores * self.score_weighing + (1 - self.score_weighing) * (ref_score + cand_score) / 2




        cand_overlap = len(overlap) / len(cand.keys()) if len(cand.keys()) != 0 else 0
        ref_overlap = len(overlap) / len(ref.keys()) if len(ref.keys()) != 0 else 0
        f1_beta_overlap = (1 + self.f1 ** 2) * (cand_overlap * ref_overlap) / (self.f1 ** 2 * cand_overlap + ref_overlap) if cand_overlap + ref_overlap != 0 else 0
        if self.overlap_type == "cand":
            overlap_cat = cand_overlap
        elif self.overlap_type == "ref":
            overlap_cat = ref_overlap
        elif self.overlap_type == "both":
            overlap_cat = (cand_overlap + ref_overlap) / 2
        elif self.overlap_type == "f1":
            overlap_cat = f1_beta_overlap
        else:
            raise ValueError("Invalid overlap_type")
        
        score = self.weighing * scores + (1 - self.weighing) * overlap_cat
        if self.apply_penalty:
            penalty = ((self.labels - len(overlap)) / self.labels) / self.penalty_score
            score = score - penalty
        score = score.item()


        return score

def get_aces_score(cands: List[str], refs: Union[List[List[str]], List[str]],
                   pipe=None, fl_weighing=False, f1_weight=0.9, 
                   overall_sbert=False, overall_sbert_weight=0.5, sbert_based_on_scores=False, **kwargs) -> float:
    if not isinstance(cands, list):
        cands = cands.tolist()
    if not isinstance(refs, list):
        refs = refs.tolist()
    cands_cas = pipe(cands, batch_size=64)
    

    if isinstance(refs[0], list):
        refs_cas = [pipe(ref, batch_size=64) for ref in refs]
    else:
        refs_cas = pipe(refs, batch_size=64)


    aces = CalculateACESScore(**kwargs)
    scores = aces(cands_cas, refs_cas)
    scores = np.array(scores)

    if overall_sbert:
        sbert_cands = model_sb.encode(cands, convert_to_tensor=True)
        sbert_refs = model_sb.encode(refs, convert_to_tensor=True)
        sbert_scores = F.normalize(sbert_cands, dim=-1) @ F.normalize(sbert_refs, dim=-1).t()
        sbert_scores = sbert_scores.mean(dim=1).detach().cpu().numpy()
        if sbert_based_on_scores and any(scores == 0.0):
            scores = np.where(scores == 0.0, sbert_scores, scores)
            scores = sbert_scores
        else:
            scores = np.add(scores * (1 - overall_sbert_weight), sbert_scores * overall_sbert_weight)


    if fl_weighing:
        fl_err = fense_eval.detect_error_sents(cands, batch_size=32)
        scores = [aces_fl-f1_weight*err*aces_fl for aces_fl,err in zip(scores, fl_err)] # Divide score by 10 if an error is found


    return scores

def cosine_similarity(input, target):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(input, target).item()


def get_text_score(all_preds_text, all_refs_text, method='sentence-bert', **kwargs):
    N = len(all_preds_text)
    K = len(all_refs_text[0])
    all_preds_text = np.array(all_preds_text, dtype=str)
    all_refs_text = np.array(all_refs_text, dtype=str)

    score = torch.zeros((N, K))
    if method == 'sentence-bert':
        preds_sb = torch.Tensor(model_sb.encode(all_preds_text))
        refs_sb = torch.Tensor(np.array([model_sb.encode(x) for x in all_refs_text]))
        # refs_sb = refs_sb.mean(dim=1)
        for i in tqdm(range(K)):
            score[:,i] = torch.Tensor([cosine_similarity(input, target) for input, target in zip(preds_sb, refs_sb[:,i])])
    elif method == 'bert-score':
        for i in tqdm(range(K)):
            P, R, F1 = scorer_bs.score(all_preds_text.tolist(), all_refs_text[:,i].tolist())
            score[:,i] = F1
    elif method == 'bleurt':
        for i in tqdm(range(K)):
            scores = scorer_brt.score(references=all_refs_text[:,i], candidates=all_preds_text)
            score[:,i] = torch.Tensor(scores).sigmoid()
    elif method.startswith("gijs/aces-roberta"):
        pipe = pipeline("token-classification", 
        model=method, aggregation_strategy=kwargs["average_strategy"], pipeline_class=ACES, 
        device=device_num if torch.cuda.is_available() else -1)
        for i in tqdm(range(K)):
            aces = get_aces_score(
                all_preds_text, 
                all_refs_text[:,i], 
                pipe=pipe,
                **kwargs
            )
            out = torch.Tensor([aces])
            score[:, i] = out
    else:
        print("Method not implemented")
        print(method)
        print()
        raise NotImplementedError
        
    out_score = score.max(dim=1)[0]

    return out_score

def get_results(config):
    metric = config['model']
    seed = 42
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    for dataset in ['audiocaps', 'clotho']:
        score, score0, score1 = {}, {}, {}

        mm_score, mm_score0, mm_score1 = {}, {}, {}

        hh_preds_text0, hh_preds_text1, hh_refs_text0, hh_refs_text1, hh_human_truth = get_former(dataset)
        mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth = get_latter(dataset)

        
        score0[metric] = get_text_score(hh_preds_text0, hh_refs_text0, metric, **config)
        score1[metric] = get_text_score(hh_preds_text1, hh_refs_text1, metric, **config)

        mm_score0[metric] = get_text_score(mm_preds_text0, mm_refs_text, metric, **config)
        mm_score1[metric] = get_text_score(mm_preds_text1, mm_refs_text, metric, **config)

        total_score0, total_score1, total_score = {}, {}, {}
        for metric in score0:
            total_score0[metric] = torch.cat([score0[metric], mm_score0[metric]])
            total_score1[metric] = torch.cat([score1[metric], mm_score1[metric]])
            total_score[metric] = total_score0[metric] - total_score1[metric]
        total_human_truth = hh_human_truth + mm_human_truth

        # metrics0, per_file_metrics0 = evaluate_metrics_from_lists(hh_preds_text0, hh_refs_text0)
        # metrics1, per_file_metrics1 = evaluate_metrics_from_lists(hh_preds_text1, hh_refs_text1)

        # expand the references (choose 4 from 5)
        # mm_preds_text0_exp = [x for x in mm_preds_text0 for i in range(5)]
        # mm_preds_text1_exp = [x for x in mm_preds_text1 for i in range(5)]
        # mm_refs_text_exp = []
        # for refs in mm_refs_text:
        #     for i in range(5):
        #         mm_refs_text_exp.append([v for k,v in enumerate(refs) if k%5!=i])

        # mm_metrics0, mm_per_file_metrics0 = evaluate_metrics_from_lists(mm_preds_text0_exp, mm_refs_text_exp)
        # mm_metrics1, mm_per_file_metrics1 = evaluate_metrics_from_lists(mm_preds_text1_exp, mm_refs_text_exp)

        def get_score_list(per_file_metric, metric):
            if metric == 'SPICE':
                return [v[metric]['All']['f'] for k,v in per_file_metric.items()]
            else:
                return [v[metric] for k,v in per_file_metric.items()]

        def shrink(arr, repeat=5):
            return np.array(arr).reshape(-1, repeat).mean(axis=1).tolist()

        # baseline_list = ['Bleu_1','Bleu_2','Bleu_3','Bleu_4','METEOR','ROUGE_L','CIDEr','SPICE','SPIDEr', '']
        # for metric in baseline_list:
        #     total_score0[metric] = torch.Tensor(get_score_list(per_file_metrics0, metric) + shrink(get_score_list(mm_per_file_metrics0, metric)))
        #     total_score1[metric] = torch.Tensor(get_score_list(per_file_metrics1, metric) + shrink(get_score_list(mm_per_file_metrics1, metric)))
        #     total_score[metric] = total_score0[metric] - total_score1[metric]

        results = []
        for metric in total_score:
            print(metric)
            tmp = print_accuracy(total_score[metric], total_human_truth)
            results.append(tmp)

        results = results[0]
        # columns=['HC', 'HC_C', 'HI', 'HI_C', 'HM', 'HM_C', 'MM', 'MM_C', 'total', 'total_C']
        results = results[::2]
        columns=['HC', 'HI', 'HM', 'MM', 'total']
        if wandb.run is not None:
            wandb.log({f"{k}_{dataset}":v for k,v in zip(columns, results)})
        else:
            # pretty print
            print(f"{dataset}:")
            for k,v in zip(columns, results):
                print(f"{k}: {v}")
            print()
    

if __name__ == "__main__":
    def main():
        wandb.init(project="aces-fl-sweep-v2")
        get_results(wandb.config)

    sweep_configuration = {
        'method': "bayes",
        "metric": {
            "goal": "maximize",
            "name": "total_clotho"
        },
        "parameters": {
            "model": {"values": ["gijs/aces-roberta-13"]},
            "fl_weighing": {"values": [True]},
            "average_strategy": {"values": ["max"]},
            "use_sbert": {"values": [True]},
            "division": {"min": 0.7, "max": 1.0},
            "f1_calc": {"values": ["max-mean"]},
            "f1_beta": {"min": 9.0, "max": 20.0}, 
            "f1": {"min": 1.0, "max": 20.0},
            "penalty_score": {"min": 700, "max": 2000},
            "f1_weight": {"min": 0.5, "max": 1.0},
            "apply_penalty": {"values": [True]},
            "overlap_type": {"values": ["both"]},
            "distance_technique": {"values": ["cosine"]},
            "use_score": {"values": ["no"]},
            "score_weighing": {"values": [0.5]},
            "overall_sbert": {"values": [False]},
            "overall_sbert_weight": {"values": [0.5]},
            "sbert_based_on_scores": {"values": [True]}
        }
    }

    sweep_id = wandb.sweep(sweep_configuration, project="aces-fl-sweep-v2")
    wandb.agent(sweep_id, function=main)

    # get_results({
    #     "model": 'gijs/aces-roberta-13', 
    #     "division": 0.9, 
    #     "fl_weighing": True, 
    #     'f1_beta': 10, 
    #     "f1": 10,
    #     "average_strategy": "max", 
    #     "use_sbert": True,
    #     "f1_calc": "max-mean",
    #     "penalty_score": 900,
    #     "apply_penalty": True,
    #     "overlap_type": "both",
    #     "f1_weight": 0.6002600452610617,
    #     "distance_technique": "cosine",
    #     "use_score": "no",
    #     "score_weighing": 0.40802690241678985,
    #     "overall_sbert": False,
    #     "overall_sbert_weight": 0.8863453667450467,
    #     "sbert_based_on_scores": True
    # })