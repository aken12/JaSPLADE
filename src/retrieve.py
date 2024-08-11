import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append('..')
sys.path.append('.')
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import json_dumps_arguments,check_dir_exist_or_build
from src.evaluate import print_res
from dataset import EncodeDataset
from collator import EncodeCollator
from models.splade import Splade
from indexing.sparse_retrieval import SparseRetrieval
from argument import get_retrieval_args

def sparse_retrieve_and_evaluate(args):
    logger.info("実行")
    model = Splade(model_type_or_dir=args.model_name)
    tokenizer = model.transformer_rep.tokenizer
    model.to(args.device)
    
    encode_collator = EncodeCollator(data_args=args,tokenizer=tokenizer)

    encode_dataset = EncodeDataset(data_args=args)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=args.batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
    )

    # get query embeddings
    qid2emb = {}
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm(encode_loader, desc="Inferencing"):
            inputs = {k: v.to(args.device) for k, v in batch[1].items()}
            batch_query_embs = model.encode(inputs,is_q=True)
            qids = batch[0] 
            batch_query_embs = batch_query_embs.cpu()
            for i, qid in enumerate(qids):
                qid2emb[qid] = batch_query_embs[i]
    
    # retrieve
    dim_voc = model.output_dim
    retriever = SparseRetrieval(args.index_dir_path, args.retrieval_output_path, dim_voc, args.top_n)
    result = retriever.retrieve(qid2emb)
    
    if os.path.exists(args.qrel_file):
        # evaluate
        eval_kwargs = {"run":result, 
                    "qrel_file": args.qrel_file, 
                    "rel_threshold": 1}
        print_res(**eval_kwargs)

        logger.info("Evaluation OK!")

if __name__ == "__main__":
    args = get_retrieval_args()
    
    # check_dir_exist_or_build([os.path.dirname(args.retrieval_output_path)], force_emptying=args.force_emptying_dir)
    
    os.makedirs(os.path.dirname(args.retrieval_output_path),exist_ok=True)
    
    paramter_file = args.retrieval_output_path.split(".")[-2]
    # json_dumps_arguments(os.path.join(args.retrieval_output_path, "parameters.txt"), args)
    json_dumps_arguments(f'{paramter_file}_param.txt', args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    sparse_retrieve_and_evaluate(args)
