#!/bin/bash

# Run commands one after another
tune run --nproc_per_node 4 full_finetune_distributed --config ./llama3_1_8B-topic.yaml
tune run --nproc_per_node 4 full_finetune_distributed --config ./llama3_1_8B-prefix.yaml
tune run --nproc_per_node 4 full_finetune_distributed --config ./llama3_1_8B-fulltext.yaml