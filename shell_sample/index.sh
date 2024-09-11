export PYTHONPATH="home/aken12_2/JaSPLADE:${PYTHONPATH}"

# model_name="aken12/splade-japanese-v3"
model_name=naver/splade-cocondenser-ensembledistil
encode_dir="encoding_$(basename $model_name)"
collection_path="" # tsv,json,jsonlどれでもよいがformatは守る必要あり

mkdir -p outputs/${encode_dir}/corpus/
mkdir -p outputs/${encode_dir}/result/

CUDA_VISIBLE_DEVICES=0 python src/index.py \
 --model_name $model_name \
 --fp16 \
 --batch_size 64 \
 --passage_max_len 180  \
 --collection_path $collection_path \
 --index_dir outputs/${encode_dir}/corpus/ \
 --local_data --dataset_shard_index 1


 