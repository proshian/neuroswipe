#!/usr/bin/env bash

export PYTHONPATH=.

python3 lib/main.py \
    --train-path data/train.jsonl \
    --test-path data/test.jsonl \
    --voc-path data/voc.txt \
    --num-workers 8 \
    --output-path result/result.csv
