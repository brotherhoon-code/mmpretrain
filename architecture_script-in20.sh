#!/bin/sh

CONFIG_NAME01="config--convmixer_v1"


CONFIG_FILE01=./configs/_scratch_/architecture/in20/${CONFIG_NAME01}.py


WORK_DIR01=./work_dir/${CONFIG_NAME01}


EXP_NAME01=${CONFIG_NAME01}


SEED=42


python tools/train.py ${CONFIG_FILE01} --work-dir ${WORK_DIR01} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME01}
