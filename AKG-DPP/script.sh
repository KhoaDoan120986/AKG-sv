#!/bin/bash
NUM_GPUS=2
MASTER_PORT=12345
OMP_NUM_THREADS=4 torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT main.py --model_id MSVD_GBased+rel+videomask