#!/bin/sh
### config_{모델}_{데이터셋}-{블럭타입}-{블럭반복}-{어그멘테이션}-{기타}

### conv-type 비교 실험
# conv

# CONFIG_NAME01="config_convmixer_in20-((dw-p)x3)x4-homo-aug"
CONFIG_NAME02="config_convmixer_in20-((p-dw)x3)x4-homo-aug"
# CONFIG_NAME03="config_convmixer_in20-((r)x3)x4-homo-aug"
# CONFIG_NAME04="config_convmixer_in20-((dw-p-p)x2)x4-homo-aug"
CONFIG_NAME05="config_convmixer_in20-((p-p-dw)x2)x4-homo-aug"


CONFIG_FILE01=./configs/_scratch_/convmixer/in20/decomp/${CONFIG_NAME01}.py
CONFIG_FILE02=./configs/_scratch_/convmixer/in20/decomp/${CONFIG_NAME02}.py
CONFIG_FILE03=./configs/_scratch_/convmixer/in20/decomp/${CONFIG_NAME03}.py
CONFIG_FILE04=./configs/_scratch_/convmixer/in20/decomp/${CONFIG_NAME04}.py
CONFIG_FILE05=./configs/_scratch_/convmixer/in20/decomp/${CONFIG_NAME05}.py


WORK_DIR01=./work_dir/${CONFIG_NAME01}
WORK_DIR02=./work_dir/${CONFIG_NAME02}
WORK_DIR03=./work_dir/${CONFIG_NAME03}
WORK_DIR04=./work_dir/${CONFIG_NAME04}
WORK_DIR05=./work_dir/${CONFIG_NAME05}


EXP_NAME01=${CONFIG_NAME01}-deit-b256
EXP_NAME02=${CONFIG_NAME02}-deit-b256
EXP_NAME03=${CONFIG_NAME03}-deit-b256
EXP_NAME04=${CONFIG_NAME04}-deit-b256
EXP_NAME05=${CONFIG_NAME05}-deit-b256



SEED=42


# python tools/train.py ${CONFIG_FILE01} --work-dir ${WORK_DIR01} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME01}
python tools/train.py ${CONFIG_FILE02} --work-dir ${WORK_DIR02} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME02}
# python tools/train.py ${CONFIG_FILE03} --work-dir ${WORK_DIR03} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME03}
# python tools/train.py ${CONFIG_FILE04} --work-dir ${WORK_DIR04} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME04}
python tools/train.py ${CONFIG_FILE05} --work-dir ${WORK_DIR05} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME05}

