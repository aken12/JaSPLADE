import numpy as np

import json
import argparse
import os 

import pytrec_eval

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluate(run, qrel_file, rel_threshold=1):
    if type(run) == str:
        with open(run, 'r' )as f:
            run_data = f.readlines()
    else:
        run_data=run
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    runs_top10 = {}
    
    
    for line in qrel_data:
        line = line.strip().split('\t')
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        qrels_ndcg[query][passage] = rel

        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.strip().split(' ')
        query = line[0]
        passage = line[2]
        rel = float(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel
    
    for q in runs:
        passages = runs[q]
        sorted_passages = sorted(passages.items(), key=lambda x: x[1], reverse=True)
        runs[q] = {passage: rel for passage, rel in sorted_passages}

    runs_top10 = {}
    for q in runs:
        top_passages = list(runs[q].items())[:10]
        runs_top10[q] = {passage: rel for passage, rel in top_passages}
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.10","recall.100","recall.1000"})
    res = evaluator.evaluate(runs)

    recall_1000_list = [v['recall_1000'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]

    res = evaluator.evaluate(runs_top10)

    mrr_list = [v['recip_rank'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.10"})
    res = evaluator.evaluate(runs)
    ndcg_10_list = [v['ndcg_cut_10'] for v in res.values()]

    res = {
            "MRR@10": round(np.average(mrr_list), 4),
            "NDCG@10": round(np.average(ndcg_10_list), 4),
            "Recall@10": round(np.average(recall_10_list), 4),
            "Recall@100": round(np.average(recall_100_list), 4),
            "Recall@1000": round(np.average(recall_1000_list), 4),
        }
    
    logger.info(res)
    return res

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--query_path',help='')
    parser.add_argument('--exp',help='',required=True)
    parser.add_argument('--run_dir',help='',default=None)
    parser.add_argument('--run',help='',default=None)
    parser.add_argument('--msmarco',help='',action='store_true')

    args = parser.parse_args()
        
    full_result = {}
    
if __name__=="__main__":
    main()

    