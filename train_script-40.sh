#!/bin/sh
### {ratio}-{block-type}-{conv-type}-{recipe}-{channel}

### conv-type 비교 실험
# conv
CONFIG_NAME01=config_resnet3_in40-R4 # check
CONFIG_NAME02=config_resnet3_in40-D4 # check
CONFIG_NAME03=config_resnet3_in40-R2D2 # check
CONFIG_NAME04=config_resnet3_in40-D2R2 # check

# ratio
CONFIG_NAME05=config_resnet3_in40-R4-ratio # check
CONFIG_NAME06=config_resnet3_in40-D4-ratio # check
CONFIG_NAME07=config_resnet3_in40-R2D2-ratio # check
CONFIG_NAME08=config_resnet3_in40-D2R2-ratio # check

# channels
CONFIG_NAME09=config_resnet3_in40-R4-channel # check
CONFIG_NAME10=config_resnet3_in40-D4-channel # check
CONFIG_NAME11=config_resnet3_in40-R2D2-channel # check
CONFIG_NAME12=config_resnet3_in40-D2R2-channel # check

# full
CONFIG_NAME13=config_resnet3_in40-R4-channel-ratio-deit
CONFIG_NAME14=config_resnet3_in40-D4-channel-ratio-deit
CONFIG_NAME15=config_resnet3_in40-R2D2-channel-ratio-deit
CONFIG_NAME16=config_resnet3_in40-D2R2-channel-ratio-deit



CONFIG_FILE01=./configs/_scratch_/in40/${CONFIG_NAME01}.py
CONFIG_FILE02=./configs/_scratch_/in40/${CONFIG_NAME02}.py
CONFIG_FILE03=./configs/_scratch_/in40/${CONFIG_NAME03}.py 
CONFIG_FILE04=./configs/_scratch_/in40/${CONFIG_NAME04}.py 
CONFIG_FILE05=./configs/_scratch_/in40/${CONFIG_NAME05}.py 
CONFIG_FILE06=./configs/_scratch_/in40/${CONFIG_NAME06}.py
CONFIG_FILE07=./configs/_scratch_/in40/${CONFIG_NAME07}.py
CONFIG_FILE08=./configs/_scratch_/in40/${CONFIG_NAME08}.py
CONFIG_FILE09=./configs/_scratch_/in40/${CONFIG_NAME09}.py
CONFIG_FILE10=./configs/_scratch_/in40/${CONFIG_NAME10}.py
CONFIG_FILE11=./configs/_scratch_/in40/${CONFIG_NAME11}.py
CONFIG_FILE12=./configs/_scratch_/in40/${CONFIG_NAME12}.py
CONFIG_FILE13=./configs/_scratch_/in40/${CONFIG_NAME13}.py
CONFIG_FILE14=./configs/_scratch_/in40/${CONFIG_NAME14}.py
CONFIG_FILE15=./configs/_scratch_/in40/${CONFIG_NAME15}.py
CONFIG_FILE16=./configs/_scratch_/in40/${CONFIG_NAME16}.py



WORK_DIR01=./work_dir/${CONFIG_NAME01}
WORK_DIR02=./work_dir/${CONFIG_NAME02}
WORK_DIR03=./work_dir/${CONFIG_NAME03}
WORK_DIR04=./work_dir/${CONFIG_NAME04}
WORK_DIR05=./work_dir/${CONFIG_NAME05}
WORK_DIR06=./work_dir/${CONFIG_NAME06}
WORK_DIR07=./work_dir/${CONFIG_NAME07}
WORK_DIR08=./work_dir/${CONFIG_NAME08}
WORK_DIR09=./work_dir/${CONFIG_NAME09}
WORK_DIR10=./work_dir/${CONFIG_NAME10}
WORK_DIR11=./work_dir/${CONFIG_NAME11}
WORK_DIR12=./work_dir/${CONFIG_NAME12}
WORK_DIR13=./work_dir/${CONFIG_NAME13}
WORK_DIR14=./work_dir/${CONFIG_NAME14}
WORK_DIR15=./work_dir/${CONFIG_NAME15}
WORK_DIR16=./work_dir/${CONFIG_NAME16}



EXP_NAME01=${CONFIG_NAME01}-sgd-b128
EXP_NAME02=${CONFIG_NAME02}-sgd-b128
EXP_NAME03=${CONFIG_NAME03}-sgd-b128
EXP_NAME04=${CONFIG_NAME04}-sgd-b128
EXP_NAME05=${CONFIG_NAME05}-sgd-b128
EXP_NAME06=${CONFIG_NAME06}-sgd-b128
EXP_NAME07=${CONFIG_NAME07}-sgd-b128
EXP_NAME08=${CONFIG_NAME08}-sgd-b128
EXP_NAME09=${CONFIG_NAME09}-sgd-b128
EXP_NAME10=${CONFIG_NAME10}-sgd-b128
EXP_NAME11=${CONFIG_NAME11}-sgd-b128
EXP_NAME12=${CONFIG_NAME12}-sgd-b128
EXP_NAME13=${CONFIG_NAME13}-sgd-b128
EXP_NAME14=${CONFIG_NAME14}-sgd-b128
EXP_NAME15=${CONFIG_NAME15}-sgd-b128
EXP_NAME16=${CONFIG_NAME16}-sgd-b128



SEED=42



python tools/train.py ${CONFIG_FILE01} --work-dir ${WORK_DIR01} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME01}
python tools/train.py ${CONFIG_FILE02} --work-dir ${WORK_DIR02} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME02}
python tools/train.py ${CONFIG_FILE03} --work-dir ${WORK_DIR03} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME03}
python tools/train.py ${CONFIG_FILE04} --work-dir ${WORK_DIR04} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME04}
python tools/train.py ${CONFIG_FILE05} --work-dir ${WORK_DIR05} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME05}
python tools/train.py ${CONFIG_FILE06} --work-dir ${WORK_DIR06} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME06}
python tools/train.py ${CONFIG_FILE07} --work-dir ${WORK_DIR07} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME07}
python tools/train.py ${CONFIG_FILE08} --work-dir ${WORK_DIR08} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME08}
python tools/train.py ${CONFIG_FILE09} --work-dir ${WORK_DIR09} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME09}
python tools/train.py ${CONFIG_FILE10} --work-dir ${WORK_DIR10} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME10}
python tools/train.py ${CONFIG_FILE11} --work-dir ${WORK_DIR11} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME11}
python tools/train.py ${CONFIG_FILE12} --work-dir ${WORK_DIR12} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME12}
