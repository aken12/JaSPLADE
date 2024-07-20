#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/13.2.0 cuda/11.2/11.2.2 cudnn/8.4/8.4.1 openjdk/11.0.22.0.7
conda activate splade

export PYTHONPATH="/home/ace14788tj/extdisk/JaSPLADE:${PYTHONPATH}"

model_name=""aken12/splade-japanese-v3""
encode_dir="encoding_$(basename $model_name)"

mkdir -p /home/ace14788tj/extdisk/JaSPLADE/outputs/${encode_dir}/corpus/
mkdir -p /home/ace14788tj/extdisk/JaSPLADE/outputs/${encode_dir}/result/

CUDA_VISIBLE_DEVICES=0 python ../../src/index.py \
 --model_name $model_name \
 --fp16 \
 --batch_size 64 \
 --passage_max_len 180  \
 --dataset_name /home/ace14788tj/aken12_2/data/msmarco/ja/mmarco_collection_japanese.tsv \
 --index_output_path /home/ace14788tj/extdisk/JaSPLADE/outputs/${encode_dir}/corpus/ \
 --local_data --dataset_shard_index 1