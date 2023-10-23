#!/usr/bin/env bash

export PYTHONPATH=.

python3 tools/viz.py \
    --path data/train.jsonl \
    --limit 10
