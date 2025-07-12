#!/bin/bash
Node=2
OMP_NUM_THREADS=4 torchrun \
  --nproc_per_node=$Node \
  --master_port=12345 \
  main.py \
  --model_id MSVD_GBased+rel+videomask \
  --attention 1 \
  --do_train