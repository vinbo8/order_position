#!/bin/bash
#SBATCH --account=project_2000582
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=16G

set -o errexit
source ~/.bashrc
conda activate ${SCRATCH}/envs/debug

ROBERTA_PATH="${PROJAPPL}/order_position/roberta/models/$1"
CONFIG_PATH="${PROJAPPL}/order_position/roberta/fairseq/examples/roberta/config/finetuning"
DATA="$(echo ${2} | awk '{print toupper($0)}')-bin"

CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train -config-dir ${CONFIG_PATH} --config-name ${2} \
	task.data=${DATA} checkpoint.restore_file=${ROBERTA_PATH}