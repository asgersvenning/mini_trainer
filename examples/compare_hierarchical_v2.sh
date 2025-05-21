#!/bin/bash

# Hierarchical
python hierarchical_train.py -i hierarchical/restructured/train -o hierarchical/restructured --name hierarchical_v3 -m efficientnet_b0 -e 50 -t --batch_size 32 -C hierarchical/restructured/hierarchical_class_index.json

# Classic
python -m mini_trainer.train -i hierarchical/restructured/train -o hierarchical/restructured --name flat_v3 -m efficientnet_b0 -e 50 -t --batch_size 32 -C hierarchical/restructured/flat_class_index.json