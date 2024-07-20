export PYTHONPATH="home/user/JaSPLADE:${PYTHONPATH}"

model_name="aken12/splade-japanese-v3"
encode_dir="encoding_$(basename $model_name)"

mkdir -p outputs/${encode_dir}/corpus/
mkdir -p outputs/${encode_dir}/result/

CUDA_VISIBLE_DEVICES=0 python src/index.py \
 --model_name $model_name \
 --fp16 \
 --batch_size 64 \
 --passage_max_len 180  \
 --dataset_name $dataset_name \
 --index_output_path outputs/${encode_dir}/corpus/ \
 --local_data --dataset_shard_index 1