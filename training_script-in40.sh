#!/bin/sh

CONFIG_NAME00="config--swinlike_resnet" 
CONFIG_NAME01="config--swinlike_resnet-ddd"
CONFIG_NAME02="config--swinlike_resnet-drr"
CONFIG_NAME03="config--swinlike_resnet-rdr"
CONFIG_NAME04="config--swinlike_resnet-rrd"
CONFIG_NAME05="..."
CONFIG_NAME06="..."
CONFIG_NAME07="..."
CONFIG_NAME08="..."


CONFIG_FILE00=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME00}.py
CONFIG_FILE01=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME01}.py
CONFIG_FILE02=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME02}.py
CONFIG_FILE03=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME03}.py
CONFIG_FILE04=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME04}.py
CONFIG_FILE05=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME05}.py
CONFIG_FILE06=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME06}.py
CONFIG_FILE07=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME07}.py
CONFIG_FILE08=./configs/_scratch_/architecture/decomposition-40/${CONFIG_NAME08}.py


WORK_DIR00=./work_dir/${CONFIG_NAME00}
WORK_DIR01=./work_dir/${CONFIG_NAME01}
WORK_DIR02=./work_dir/${CONFIG_NAME02}
WORK_DIR03=./work_dir/${CONFIG_NAME03}
WORK_DIR04=./work_dir/${CONFIG_NAME04}
WORK_DIR05=./work_dir/${CONFIG_NAME05}
WORK_DIR06=./work_dir/${CONFIG_NAME06}
WORK_DIR07=./work_dir/${CONFIG_NAME07}
WORK_DIR08=./work_dir/${CONFIG_NAME08}


EXP_NAME00=${CONFIG_NAME00}
EXP_NAME01=${CONFIG_NAME01}
EXP_NAME02=${CONFIG_NAME02}
EXP_NAME03=${CONFIG_NAME03}
EXP_NAME04=${CONFIG_NAME04}
EXP_NAME05=${CONFIG_NAME05}
EXP_NAME06=${CONFIG_NAME06}
EXP_NAME07=${CONFIG_NAME07}
EXP_NAME08=${CONFIG_NAME08}


SEED=42


python tools/train.py ${CONFIG_FILE00} --work-dir ${WORK_DIR00} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME00}
python tools/train.py ${CONFIG_FILE01} --work-dir ${WORK_DIR01} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME01}
python tools/train.py ${CONFIG_FILE02} --work-dir ${WORK_DIR02} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME02}
python tools/train.py ${CONFIG_FILE03} --work-dir ${WORK_DIR03} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME03}
python tools/train.py ${CONFIG_FILE04} --work-dir ${WORK_DIR04} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME04}
# python tools/train.py ${CONFIG_FILE05} --work-dir ${WORK_DIR05} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME05}
# python tools/train.py ${CONFIG_FILE06} --work-dir ${WORK_DIR06} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME06}
# python tools/train.py ${CONFIG_FILE07} --work-dir ${WORK_DIR07} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME07}
# python tools/train.py ${CONFIG_FILE08} --work-dir ${WORK_DIR08} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME08}
