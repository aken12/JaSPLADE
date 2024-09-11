from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset 

import random

import logging
import json
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
class EncodeDataset(Dataset):
    def __init__(self,data_args):
        self.data_args = data_args

        logger.info(self.data_args.collection_path)
        if self.data_args.local_data:
            self.encode_data = {"query_id": [],"query": []} if self.data_args.encode_is_query \
                               else {"docid": [],"text": []} 
            
            if self.data_args.encode_is_query:
                if self.data_args.use_pseudo_doc:
                    self.encode_data["pseudo_doc"] = []
            else:
                if self.data_args.title:
                    self.encode_data["title"] = []
            
            with open(self.data_args.collection_path) as fr:
                data_format = self.data_args.collection_path.split('.')[-1]
                if self.data_args.ignore_first_line:
                    next(fr)
                if data_format == "tsv":
                    for line in fr:
                        line = line.strip().split('\t')

                        if self.data_args.encode_is_query:
                            self.encode_data["query_id"].append(line[0])
                            self.encode_data["query"].append(line[1])
                        else:
                            self.encode_data["docid"].append(line[0])
                            self.encode_data["text"].append(line[1])
                            if self.data_args.title:
                                self.encode_data['title'].append(line[2])           
                else:
                    if data_format == "json":
                        logger.info("Processing : json")   
                        json_data = json.load(fr)
                    else:      
                        json_data = []
                        logger.info("Processing : jsonl")            
                        for line in tqdm(fr):
                            line = json.loads(line)
                            json_data.append(line)
                            
                    for line in json_data:
                        if self.data_args.encode_is_query:
                            self.encode_data["query_id"].append(line["query_id"])
                            self.encode_data["query"].append(line["query"])
                            if self.data_args.use_pseudo_doc:
                                self.encode_data["pseudo_doc"].append(line["pseudo_doc"])
                        else:
                            self.encode_data["docid"].append(line["id"])
                            self.encode_data["text"].append(line["contents"])
                            if self.data_args.title:
                                title = line["title"] if line["title"] != None else ""
                                self.encode_data['title'].append(title)

            self.encode_data = HFDataset.from_dict(self.encode_data)

        else:
            
            self.encode_data = load_dataset(
                self.data_args.collection_path,
                self.data_args.dataset_config,
                split=self.data_args.dataset_split,
            )


        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
    
        logger.info(len(self.encode_data))

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_text = text['query']
            if self.data_args.use_pseudo_doc:
                pseudo_doc = text['pseudo_doc']
                formated_text = f"{formated_text} {pseudo_doc}"
            else:
                formated_text = formated_text
        else:
            text_id = text['docid']
            if self.data_args.title:
                formated_text = text['title'] + ' ' + text['text']
                formated_text = formated_text.strip()
            else:
                formated_text = text['text']
        return text_id, formated_text