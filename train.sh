#!/usr/bin/env bash
# D4 training launcher. Run from the repository root:
#   bash experiment/exp4/D4/train.sh experiment/exp4/D4/options/Deraining_Restormer_30k_WFFN_v1.yml [NGPU] [PORT]
# NGPU must equal num_gpu in the yml (default 2). Outputs go to experiment/exp4/D4/experiments/<name>.
set -e

CONFIG=$1
NGPU=${2:-2}
PORT=${3:-4321}

if [ -z "$CONFIG" ]; then
    echo "usage: bash experiment/exp4/D4/train.sh <option.yml> [NGPU] [PORT]"
    exit 1
fi

torchrun --nproc_per_node=$NGPU --master_port=$PORT basicsr/train.py -opt $CONFIG --launcher pytorch
