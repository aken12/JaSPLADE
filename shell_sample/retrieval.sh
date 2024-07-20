export PYTHONPATH="home/user/JaSPLADE:${PYTHONPATH}"

exp=msmarco
model_name="naver/splade-cocondenser-ensembledistil"
encode_dir="encoding_$(basename $model_name)"

CUDA_VISIBLE_DEVICES=0 python src/retrieve.py \
 --model_name $model_name \
 --fp16 \
 --batch_size 16 \
 --query_max_len 256 --encode_is_query \
 --dataset_name $dataset_name \
 --index_dir_path ${encode_dir}/corpus/ \
 --local_data --retrieval_output_path ${encode_dir}/result_${exp}/ \
 --top_n 1000 --rel_threshold 1 --force_emptying_dir --qrel_file $qrel_file