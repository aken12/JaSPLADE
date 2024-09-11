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
exp="example_run"
query_path="/home/ace14788tj/aken12_2/data/msmarco/msmarco/queries.eval.small.tsv"

CUDA_VISIBLE_DEVICES=0 python src/retrieve.py \
 --model_name $model_name \
 --fp16 \
 --batch_size 16 --query_path $query_path \
 --query_max_len 256 --encode_is_query \
 --index_dir outputs/${encode_dir}/corpus/ \
 --local_data --retrieval_output_path outputs/${encode_dir}/result_${exp}/$(basename $model_name)_${exp}.txt \
 --top_n 100 --rel_threshold 1