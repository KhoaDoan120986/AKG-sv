#!/bin/bash
OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 --master_port=12345 main.py --model_id MSVD_GBased+rel+videomask