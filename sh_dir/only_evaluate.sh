#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=8:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ~/.bashrc
module load gcc/13.2.0 cuda/11.2/11.2.2 cudnn/8.4/8.4.1 openjdk/11.0.22.0.7
conda activate splade

export PYTHONPATH="/home/ace14788tj/extdisk:${PYTHONPATH}"

exp=q2d
model_name="naver/splade-cocondenser-ensembledistil"
encode_dir="encoding_$(basename $model_name)"

CUDA_VISIBLE_DEVICES=0 python ../../only_evaluate.py \
 --model_name $model_name \
 --fp16 \
 --batch_size 16 \
 --query_max_len 256 --encode_is_query \
 --dataset_name /home/ace14788tj/extdisk/data/msmarco/query/q2d_query.json \
 --index_dir_path ${encode_dir}/corpus/ \
 --local_data --retrieval_output_path ${encode_dir}/result/ \
 --top_n 1000 --rel_threshold 1 --qrel_file /home/ace14788tj/aken12_2/data/msmarco/msmarco/qrels.dev.small.tsv