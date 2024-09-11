#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -cwd
#$ -p -400

source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/13.2.0 cuda/11.2/11.2.2 cudnn/8.4/8.4.1 openjdk/17.0.12.0.7
conda activate splade

export PYTHONPATH="home/aken12_2/JaSPLADE:${PYTHONPATH}"

# model_name="aken12/splade-japanese-v3"
model_name=naver/splade-cocondenser-ensembledistil
encode_dir="encoding_$(basename $model_name)"
collection_path="/home/ace14788tj/aken12_2/data/msmarco/msmarco/queries.eval.small.tsv"

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


 