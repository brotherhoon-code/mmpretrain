#!/bin/sh
CONFIG_NAME="convnextT"
CONFIG_FILE=./configs/_scratch_/!AUG_IN20/${CONFIG_NAME}.py
WORK_DIR=./work_dir/${CONFIG_NAME}
EXP_NAME=${CONFIG_NAME}

SEED=42

python tools/train.py ${CONFIG_FILE} --work-dir ${WORK_DIR} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME}
