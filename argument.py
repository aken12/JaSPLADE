import argparse

def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, required=True)
    parser.add_argument('--model_name', default='naver/splade-v3', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--query_max_len', default=32, type=int)
    parser.add_argument('--passage_max_len', default=180, type=int)
    parser.add_argument('--encode_is_query', action='store_true')
    parser.add_argument('--dataset_shard_index', default=0, type=int)
    parser.add_argument('--dataset_number_of_shards', default=1, type=int)
    parser.add_argument('--dataloader_num_workers', default=10, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--local_data', action='store_true')
    parser.add_argument('--dataset_split', type=str)
    parser.add_argument('--dataset_config', type=str)
    parser.add_argument('--lower_text', action='store_true')
    parser.add_argument('--ignore_first_line', action='store_true')
    return parser

def get_index_args():
    parser = get_base_args()
    parser.description = 'Indexing arguments'
    parser.add_argument('--title', action='store_true')
    parser.add_argument('--collection_path', type=str)
    return parser.parse_args()

def get_retrieval_args():
    parser = get_base_args()
    parser.description = 'Retrieval arguments'
    parser.add_argument('--qrel_file', type=str,default=None)
    parser.add_argument('--use_pseudo_doc', action='store_true')
    parser.add_argument('--rel_threshold', type=int, required=True)
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--retrieval_output_path', type=str, required=True)
    return parser.parse_args()

