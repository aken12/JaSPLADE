import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append('..')
sys.path.append('.')
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import json_dumps_arguments,check_dir_exist_or_build
from src.evaluate import print_res

def evaluate(args):    
    # evaluate
    eval_kwargs = {"run": os.path.join(args.retrieval_output_path,"run.txt"),
                   "qrel_file": args.qrel_file, 
                   "rel_threshold": 1}
    print_res(**eval_kwargs)

    logger.info("Evaluation OK!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument('--qrel_file', dest='qrel_file',type=str)

    args = parser.parse_args()

    # json_dumps_arguments(os.path.join(args.retrieval_output_path, "parameters.txt"), args)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = device
    args.start_running_time = time.asctime(time.localtime(time.time()))
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    evaluate(args)
