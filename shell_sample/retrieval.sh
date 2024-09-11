export PYTHONPATH="home/user/JaSPLADE:${PYTHONPATH}"

# model_name="aken12/splade-japanese-v3"
model_name=naver/splade-cocondenser-ensembledistil
encode_dir="encoding_$(basename $model_name)"
exp="example_run"
query_path="" # tsv,json,jsonlどれでもよいがformatは守る必要あり
qrel_path="" # TREC format

CUDA_VISIBLE_DEVICES=0 python src/retrieve.py \
 --model_name $model_name \
 --fp16 \
 --batch_size 16 --query_path $query_path \
 --query_max_len 256 --encode_is_query \
 --index_dir outputs/${encode_dir}/corpus/ \
 --local_data --retrieval_output_path outputs/${encode_dir}/result_${exp}/$(basename $model_name)_${exp}.txt \
 --top_n 100 --rel_threshold 1 --qrel_path 