import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,1"
import sys
sys.path.insert(0, os.getcwd()) 
from src.aces import get_aces_score, ACES
import json
import numpy as np
import pandas as pd
import json 
import torch
from tqdm import tqdm
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers.pipelines.token_classification import AggregationStrategy
from transformers import pipeline
from bert_score import BERTScorer
from bleurt import score as bleurt_score
from evaluation.eval_metrics import evaluate_metrics_from_lists
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataloader import get_former, get_latter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device_num = 1

model_sb = SentenceTransformer('paraphrase-TinyBERT-L6-v2', device=device2)
model_sb.eval()

scorer_bs = BERTScorer(model_type='ramsrigouthamg/t5_paraphraser', lang="en", rescale_with_baseline=False, device=device2)

scorer_brt = bleurt_score.BleurtScorer()

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
        for i in tqdm(range(K)):
            aces = get_aces_score(
                all_preds_text, 
                all_refs_text[:,i], 
                model="base" if 'base' in method else 'large',
                **kwargs
            )
            out = torch.Tensor([aces])
            score[:, i] = out
    else:
        raise NotImplementedError
        
    out_score = score.max(dim=1)[0]

    return out_score

def get_accuracy(machine_score, human_score, threshold=0):
    cnt = 0

    N = np.sum([x!=0 for x in human_score]) if threshold==0 else len(human_score)
    for i, (ms, hs) in enumerate(zip(machine_score, human_score)):
        if ms*hs > 0 or abs(ms-hs) < threshold:
            cnt += 1
    accuracy = cnt / N
    # conf = 1.64 * np.sqrt(accuracy * (1 - accuracy) / N)
    return accuracy

def print_accuracy(machine_score, human_score):
    results = []
    for i, facet in enumerate(['HC', 'HI', 'HM', 'MM']):
        if facet != 'MM':
            sub_score = machine_score[i*250:(i+1)*250]
            sub_truth = human_score[i*250:(i+1)*250]
        else:
            sub_score = machine_score[i*250:]
            sub_truth = human_score[i*250:]
        acc = get_accuracy(sub_score, sub_truth)
        results.append(round(acc*100, 1))
        print(facet,  "%.1f" % (acc*100))
    acc = get_accuracy(machine_score, human_score)
    results.append(round(acc*100, 1))
    print("total acc: %.1f" % (acc*100))
    return results

if __name__ == '__main__':
    for dataset in ['audiocaps', 'clotho']:
        score, score0, score1 = {}, {}, {}

        mm_score, mm_score0, mm_score1 = {}, {}, {}

        hh_preds_text0, hh_preds_text1, hh_refs_text0, hh_refs_text1, hh_human_truth = get_former(dataset)
        mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth = get_latter(dataset)

        dir_paths = ["output"]
        metrics = ['sentence-bert', 'bleurt', 'bert-score', "gijs/aces-roberta-13"]
        # metrics = ["gijs/aces-roberta-13"]


        print("metrics:", metrics)
        for metric in tqdm(metrics):
            score0[metric] = get_text_score(hh_preds_text0, hh_refs_text0, metric)
            score1[metric] = get_text_score(hh_preds_text1, hh_refs_text1, metric)

            mm_score0[metric] = get_text_score(mm_preds_text0, mm_refs_text, metric)
            mm_score1[metric] = get_text_score(mm_preds_text1, mm_refs_text, metric)
        
        total_score0, total_score1, total_score = {}, {}, {}
        for metric in score0:
            total_score0[metric] = torch.cat([score0[metric], mm_score0[metric]])
            total_score1[metric] = torch.cat([score1[metric], mm_score1[metric]])
            total_score[metric] = total_score0[metric] - total_score1[metric]
        total_human_truth = hh_human_truth + mm_human_truth

        metrics0, per_file_metrics0 = evaluate_metrics_from_lists(hh_preds_text0, hh_refs_text0)
        metrics1, per_file_metrics1 = evaluate_metrics_from_lists(hh_preds_text1, hh_refs_text1)

        # expand the references (choose 4 from 5)
        mm_preds_text0_exp = [x for x in mm_preds_text0 for i in range(5)]
        mm_preds_text1_exp = [x for x in mm_preds_text1 for i in range(5)]
        mm_refs_text_exp = []
        for refs in mm_refs_text:
            for i in range(5):
                mm_refs_text_exp.append([v for k,v in enumerate(refs) if k%5!=i])

        mm_metrics0, mm_per_file_metrics0 = evaluate_metrics_from_lists(mm_preds_text0_exp, mm_refs_text_exp)
        mm_metrics1, mm_per_file_metrics1 = evaluate_metrics_from_lists(mm_preds_text1_exp, mm_refs_text_exp)

        def get_score_list(per_file_metric, metric):
            if metric == 'SPICE':
                return [v[metric]['All']['f'] for k,v in per_file_metric.items()]
            else:
                return [v[metric] for k,v in per_file_metric.items()]

        def shrink(arr, repeat=5):
            return np.array(arr).reshape(-1, repeat).mean(axis=1).tolist()

        baseline_list = ['Bleu_1','Bleu_2','Bleu_3','Bleu_4','METEOR','ROUGE_L','CIDEr','SPICE','SPIDEr']
        for metric in baseline_list:
            total_score0[metric] = torch.Tensor(get_score_list(per_file_metrics0, metric) + shrink(get_score_list(mm_per_file_metrics0, metric)))
            total_score1[metric] = torch.Tensor(get_score_list(per_file_metrics1, metric) + shrink(get_score_list(mm_per_file_metrics1, metric)))
            total_score[metric] = total_score0[metric] - total_score1[metric]

        results = []
        all_results = []
        for metric in total_score:
            print(metric)
            tmp = print_accuracy(total_score[metric], total_human_truth)
            results.append(tmp)
            for i, facet in enumerate(['HC', 'HI', 'HM', 'MM']):
                if facet != 'MM':
                    sub_score = total_score[metric][i*250:(i+1)*250]
                    sub_truth = total_human_truth[i*250:(i+1)*250]
                else:
                    sub_score = total_score[metric][i*250:]
                    sub_truth = total_human_truth[i*250:]
                # N = np.sum([x!=0 for x in sub_truth])
                for i, (ms, hs) in enumerate(zip(sub_score, sub_truth)):
                    if facet != 'MM':
                        all_results.append({
                            'metric': metric,
                            'facet': facet,
                            'ms': ms.item(),
                            'hs': hs,
                            'score': (ms*hs > 0 or abs(ms-hs) < 0).item(),
                            'pred0': hh_preds_text0[i],
                            'ref0_0': hh_refs_text0[i][0],
                            'ref0_1': hh_refs_text0[i][1],
                            'ref0_2': hh_refs_text0[i][2],
                            'ref0_3': hh_refs_text0[i][3],
                            'pred1': hh_preds_text1[i],
                            'ref1_0': hh_refs_text1[i][0],
                            'ref1_1': hh_refs_text1[i][1],
                            'ref1_2': hh_refs_text1[i][2],
                            'ref1_3': hh_refs_text1[i][3],
                        })
                    else:
                        all_results.append({
                            'metric': metric,
                            'facet': facet,
                            'ms': ms.item(),
                            'hs': hs,
                            'score': (ms*hs > 0 or abs(ms-hs) < 0).item(),
                            'pred0': mm_preds_text0[i],
                            'ref0_0': mm_refs_text[i][0],
                            'ref0_1': mm_refs_text[i][1],
                            'ref0_2': mm_refs_text[i][2],
                            'ref0_3': mm_refs_text[i][3],
                            'pred1': mm_preds_text1[i],
                            'ref1_0': mm_refs_text[i][0],
                            'ref1_1': mm_refs_text[i][1],
                            'ref1_2': mm_refs_text[i][2],
                            'ref1_3': mm_refs_text[i][3],
                        })



        df = pd.DataFrame(results, 
        columns=['HC', 'HI', 'HM', 'MM', 'total'])
        df.index = [x for x in total_score]
        df.to_csv('./evaluation/fense_experiment/experiment/data/results_all_{}.csv'.format(dataset))


        df = pd.DataFrame(all_results)
        df.to_csv('./evaluation/fense_experiment/experiment/data/results_all_{}_all.csv'.format(dataset), index=False)
        

        score_penalty = {}
        thres = 0.9
        coef = 0.9

        # for method in total_score:
        #     score_penalty[method] = [s1-s1*coef*(p1>thres)-(s2-s2*coef*(p2>thres)) for s1,s2,p1,p2 in zip(total_score0[method],total_score1[method],probs0[:,-1],probs1[:,-1])]

        # fluency_results = []
        # for method in score_penalty:
        #     print(method)
        #     tmp = print_accuracy(score_penalty[method], total_human_truth)
        #     fluency_results.append(tmp)

        # df = pd.DataFrame(fluency_results, columns=['HC', 'HI', 'HM', 'MM', 'total'])
        # df.index = [x for x in total_score]
        # df.to_csv('./evaluation/fense-experiment/experiment/data/results_all_{}_fluency.csv'.format(dataset))

