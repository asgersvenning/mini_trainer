#!/bin/bash

# micromamba run -n mini_trainer python -m mini_trainer.train \
#     -i hierarchical/gmo_traits \
#     -o hierarchical \
#     -n flat_v2 \
#     -t \
#     -m efficientnet_b0 \
#     -C hierarchical/gmo_traits_class_index.json \
#     -D hierarchical/gmo_traits_data_index.json \
#     -e 25 \
#     --batch_size 32 \
#     --warmup_epochs 1

micromamba run -n mini_trainer python hierarchical_guillaume.py