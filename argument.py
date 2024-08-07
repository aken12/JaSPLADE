import argparse

def get_index_args():
    parser = argparse.ArgumentParser(description='Indexing arguments')
    parser.add_argument('--index_output_path', dest='index_output_path',default=None,type=str,required=True)
    parser.add_argument('--model_name', dest='model_name',default='naver/splade-v3',type=str,required=True)
    parser.add_argument('--batch_size', dest='batch_size',default=None,type=int,required=True)
    parser.add_argument('--dataset_name', dest='dataset_name',default=None,type=str,required=True)
    parser.add_argument('--collection_path', dest='collection_path',default=None,type=str)
    parser.add_argument('--device', dest='device',default='cuda',type=str)
    parser.add_argument('--query_max_len', dest='query_max_len',default=32,type=int)
    parser.add_argument('--passage_max_len', dest='passage_max_len',default=180,type=int)
    parser.add_argument('--encode_is_query', dest='encode_is_query',action='store_true')
    parser.add_argument('--dataset_shard_index', dest='dataset_shard_index',default=0,type=int)
    parser.add_argument('--dataset_number_of_shards', dest='dataset_number_of_shards',default=1,type=int)
    parser.add_argument('--dataloader_num_workers', dest='dataloader_num_workers',default=10,type=int)
    parser.add_argument('--fp16', dest='fp16',action='store_true')
    parser.add_argument('--title', dest='title',action='store_true')
    parser.add_argument('--local_data', dest='local_data',action='store_true')
    parser.add_argument('--use_pseudo_doc', dest='use_pseudo_doc',action='store_true')
    parser.add_argument('--normalize', dest='normalize',action='store_true')
    parser.add_argument('--dataset_split', dest='dataset_split',default=None,type=str)
    parser.add_argument('--dataset_config', dest='dataset_config',default=None,type=str)
    parser.add_argument('--lower_text', dest='lower_text',action='store_true')
    
    return parser.parse_args()

def get_retrieval_args():
    parser = argparse.ArgumentParser(description='Retrieval arguments')
    parser.add_argument('--index_dir_path', dest='index_dir_path',default=None,type=str,required=True)
    parser.add_argument('--model_name', dest='model_name',default='naver/splade-v3',type=str,required=True)
    parser.add_argument('--batch_size', dest='batch_size',default=None,type=int,required=True)
    parser.add_argument('--dataset_name', dest='dataset_name',default=None,type=str,required=True)
    parser.add_argument('--device', dest='device',default='cuda',type=str)
    parser.add_argument('--qrel_file', dest='qrel_file',type=str)
    parser.add_argument('--query_max_len', dest='query_max_len',default=32,type=int)
    parser.add_argument('--passage_max_len', dest='passage_max_len',default=180,type=int)
    parser.add_argument('--fp16', dest='fp16',action='store_true')
    parser.add_argument('--encode_is_query', dest='encode_is_query',action='store_true')
    parser.add_argument('--local_data', dest='local_data',action='store_true')
    parser.add_argument('--title', dest='title',action='store_true')
    parser.add_argument('--use_pseudo_doc', dest='use_pseudo_doc',action='store_true')
    parser.add_argument('--dataset_split', dest='dataset_split',default=None,type=str)
    parser.add_argument('--dataset_config', dest='dataset_config',default=None,type=str)
    parser.add_argument('--dataset_shard_index', dest='dataset_shard_index',default=0,type=int)
    parser.add_argument('--dataset_number_of_shards', dest='dataset_number_of_shards',default=1,type=int)
    parser.add_argument('--dataloader_num_workers', dest='dataloader_num_workers',default=10,type=int)
    parser.add_argument("--rel_threshold", type=int, required=True, help="CAsT-20: 2, Others: 1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_data_percent", type=float, default=1.0, help="Percent of samples to use. Faciliating the debugging.")
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--retrieval_output_path", type=str, required=True)
    parser.add_argument("--force_emptying_dir", action="store_true", help="Force to empty the (output) dir.")
    return parser.parse_args()
