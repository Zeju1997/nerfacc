##!/usr/bin/bash

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate nerfacc

# module load cuda/11.3

# seeds=(10094 16734 20058 26284 27026)

python examples/train_mlp_nerf.py --train_split train --scene lego