#!/bin/sh

CONFIG_NAME00="config--resnet"


CONFIG_FILE00=./configs/_scratch_/architecture/in20/${CONFIG_NAME00}.py


WORK_DIR00=./work_dir/${CONFIG_NAME00}


EXP_NAME00=${CONFIG_NAME00}


SEED=42


python tools/train.py ${CONFIG_FILE00} --work-dir ${WORK_DIR00} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME00}