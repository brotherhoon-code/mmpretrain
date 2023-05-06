#!/bin/sh
### config_{모델}_{데이터셋}-{블럭타입}-{블럭반복}-{어그멘테이션}-{기타}

### conv-type 비교 실험
# conv
# CONFIG_NAME00="resnet18_8xb16_cifar10" 
# CONFIG_NAME01="config_convmixer_in20-(dw-p)x4-homo" 
# CONFIG_NAME02="config_convmixer_in20-(r-p)x4-homo"

# CONFIG_NAME03="config_convmixer_in20-(dw-p)x4-homo-aug"
# CONFIG_NAME04="config_convmixer_in20-(r-p)x4-homo-aug"

# CONFIG_NAME05="config_convmixer_in20-(dw-p)x4-homo-aug-300ep" 
# CONFIG_NAME06="config_convmixer_in20-(r-p)x4-homo-aug-300ep" 

CONFIG_NAME07="config_convmixer_in20-(dw-p-p)x4-homo-aug" 
CONFIG_NAME08="config_convmixer_in20-(r-p-p)x4-homo-aug" 
CONFIG_NAME09="config_convmixer_in20-(p-p-dw)x4-homo-aug" 
CONFIG_NAME10="config_convmixer_in20-(p-dw-p)x4-homo-aug" 

# CONFIG_NAME11="config_convmixer_in20-(p-p-r)x4-homo-aug"
# CONFIG_NAME08="config_convmixer_in20-(r-p-p)x4-homo-aug" 
# CONFIG_NAME12="config_convmixer_in20-(p-r-p)x4-homo-aug"
# CONFIG_NAME13="config_convmixer_in20-(r)x4-homo-aug"

# CONFIG_NAME14="config_convmixer_in20-(r1)x1(dw-p)x3-homo-aug"
# CONFIG_NAME15="config_convmixer_in20-(r2)x1(dw-p)x3-homo-aug"
# CONFIG_NAME16="config_convmixer_in20-(r3)x1(dw-p)x3-homo-aug"
# CONFIG_NAME17="config_convmixer_in20-(r4)x1(dw-p)x3-homo-aug"

# CONFIG_NAME18="config_convmixer_in20-(r-dwp-dwp)x4-inhomo-aug"
# CONFIG_NAME19="config_convmixer_in20-(dwp-r-dwp)x4-inhomo-aug"
# CONFIG_NAME20="config_convmixer_in20-(dwp-dwp-r)x4-inhomo-aug"

# CONFIG_NAME21="config_convmixer_in20-(p-dw)x4-homo-aug"
# CONFIG_NAME22="config_convmixer_in20-(ppdw-r-dwp)x4-inhomo-aug"
# CONFIG_NAME23="config_convmixer_in20-(ppdw-ppdw-r)x4-inhomo-aug"

CONFIG_NAME24="config_convmixer_in20-((r)x2)x4-homo-aug"
# CONFIG_NAME25="config_convmixer_in20-((r-dwp-dwp)x2)x4-inhomo-aug"
# CONFIG_NAME26="config_convmixer_in20-((dwp-dwp-r)x2)x4-inhomo-aug"
# CONFIG_NAME27="config_convmixer_in20-((dwp-r-dwp)x2)x4-inhomo-aug"




CONFIG_FILE00=./configs/resnet/${CONFIG_NAME00}.py
CONFIG_FILE01=./configs/_scratch_/convmixer/in20/${CONFIG_NAME01}.py
CONFIG_FILE02=./configs/_scratch_/convmixer/in20/${CONFIG_NAME02}.py
CONFIG_FILE03=./configs/_scratch_/convmixer/in20/${CONFIG_NAME03}.py
CONFIG_FILE04=./configs/_scratch_/convmixer/in20/${CONFIG_NAME04}.py
CONFIG_FILE05=./configs/_scratch_/convmixer/in20/${CONFIG_NAME05}.py
CONFIG_FILE06=./configs/_scratch_/convmixer/in20/${CONFIG_NAME06}.py
CONFIG_FILE07=./configs/_scratch_/convmixer/in20/${CONFIG_NAME07}.py
CONFIG_FILE08=./configs/_scratch_/convmixer/in20/${CONFIG_NAME08}.py
CONFIG_FILE09=./configs/_scratch_/convmixer/in20/${CONFIG_NAME09}.py
CONFIG_FILE10=./configs/_scratch_/convmixer/in20/${CONFIG_NAME10}.py
CONFIG_FILE11=./configs/_scratch_/convmixer/in20/${CONFIG_NAME11}.py
CONFIG_FILE12=./configs/_scratch_/convmixer/in20/${CONFIG_NAME12}.py
CONFIG_FILE13=./configs/_scratch_/convmixer/in20/${CONFIG_NAME13}.py
CONFIG_FILE14=./configs/_scratch_/convmixer/in20/${CONFIG_NAME14}.py
CONFIG_FILE15=./configs/_scratch_/convmixer/in20/${CONFIG_NAME15}.py
CONFIG_FILE16=./configs/_scratch_/convmixer/in20/${CONFIG_NAME16}.py
CONFIG_FILE17=./configs/_scratch_/convmixer/in20/${CONFIG_NAME17}.py
CONFIG_FILE18=./configs/_scratch_/convmixer/in20/${CONFIG_NAME18}.py
CONFIG_FILE19=./configs/_scratch_/convmixer/in20/${CONFIG_NAME19}.py
CONFIG_FILE20=./configs/_scratch_/convmixer/in20/${CONFIG_NAME20}.py
CONFIG_FILE21=./configs/_scratch_/convmixer/in20/${CONFIG_NAME21}.py
CONFIG_FILE22=./configs/_scratch_/convmixer/in20/${CONFIG_NAME22}.py
CONFIG_FILE23=./configs/_scratch_/convmixer/in20/${CONFIG_NAME23}.py
CONFIG_FILE24=./configs/_scratch_/convmixer/in20/${CONFIG_NAME24}.py
CONFIG_FILE25=./configs/_scratch_/convmixer/in20/${CONFIG_NAME25}.py
CONFIG_FILE26=./configs/_scratch_/convmixer/in20/${CONFIG_NAME26}.py
CONFIG_FILE27=./configs/_scratch_/convmixer/in20/${CONFIG_NAME27}.py


WORK_DIR00=./work_dir/${CONFIG_NAME00}
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
WORK_DIR17=./work_dir/${CONFIG_NAME17}
WORK_DIR18=./work_dir/${CONFIG_NAME18}
WORK_DIR19=./work_dir/${CONFIG_NAME19}
WORK_DIR20=./work_dir/${CONFIG_NAME20}
WORK_DIR21=./work_dir/${CONFIG_NAME21}
WORK_DIR22=./work_dir/${CONFIG_NAME22}
WORK_DIR23=./work_dir/${CONFIG_NAME23}
WORK_DIR24=./work_dir/${CONFIG_NAME24}
WORK_DIR25=./work_dir/${CONFIG_NAME25}
WORK_DIR26=./work_dir/${CONFIG_NAME26}
WORK_DIR27=./work_dir/${CONFIG_NAME27}


EXP_NAME00=${CONFIG_NAME00}-test
EXP_NAME01=${CONFIG_NAME01}-deit-b256
EXP_NAME02=${CONFIG_NAME02}-deit-b256
EXP_NAME03=${CONFIG_NAME03}-deit-b256
EXP_NAME04=${CONFIG_NAME04}-deit-b256
EXP_NAME05=${CONFIG_NAME05}-deit-b256
EXP_NAME06=${CONFIG_NAME06}-deit-b256
EXP_NAME07=${CONFIG_NAME07}-deit-b256
EXP_NAME08=${CONFIG_NAME08}-deit-b256
EXP_NAME09=${CONFIG_NAME09}-deit-b256
EXP_NAME10=${CONFIG_NAME10}-deit-b256
EXP_NAME11=${CONFIG_NAME11}-deit-b256
EXP_NAME12=${CONFIG_NAME12}-deit-b256
EXP_NAME13=${CONFIG_NAME13}-deit-b256
EXP_NAME14=${CONFIG_NAME14}-deit-b256
EXP_NAME15=${CONFIG_NAME15}-deit-b256
EXP_NAME16=${CONFIG_NAME16}-deit-b256
EXP_NAME17=${CONFIG_NAME17}-deit-b256
EXP_NAME18=${CONFIG_NAME18}-deit-b256
EXP_NAME19=${CONFIG_NAME19}-deit-b256
EXP_NAME20=${CONFIG_NAME20}-deit-b256
EXP_NAME21=${CONFIG_NAME21}-deit-b256
EXP_NAME22=${CONFIG_NAME22}-deit-b256
EXP_NAME23=${CONFIG_NAME23}-deit-b256
EXP_NAME24=${CONFIG_NAME24}-deit-b256
EXP_NAME25=${CONFIG_NAME25}-deit-b256
EXP_NAME26=${CONFIG_NAME26}-deit-b256
EXP_NAME27=${CONFIG_NAME27}-deit-b256


SEED=42


# python tools/train.py ${CONFIG_FILE00}
# python tools/train.py ${CONFIG_FILE01} --work-dir ${WORK_DIR01} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME01}
# python tools/train.py ${CONFIG_FILE02} --work-dir ${WORK_DIR02} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME02}
# python tools/train.py ${CONFIG_FILE03} --work-dir ${WORK_DIR03} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME03}
# python tools/train.py ${CONFIG_FILE04} --work-dir ${WORK_DIR04} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME04}
# python tools/train.py ${CONFIG_FILE05} --work-dir ${WORK_DIR05} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME05}
# python tools/train.py ${CONFIG_FILE06} --work-dir ${WORK_DIR06} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME06}
# python tools/train.py ${CONFIG_FILE07} --work-dir ${WORK_DIR07} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME07}
# python tools/train.py ${CONFIG_FILE08} --work-dir ${WORK_DIR08} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME08}
# python tools/train.py ${CONFIG_FILE09} --work-dir ${WORK_DIR09} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME09}
# python tools/train.py ${CONFIG_FILE10} --work-dir ${WORK_DIR10} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME10}
# python tools/train.py ${CONFIG_FILE11} --work-dir ${WORK_DIR11} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME11}
# python tools/train.py ${CONFIG_FILE12} --work-dir ${WORK_DIR12} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME12}
# python tools/train.py ${CONFIG_FILE13} --work-dir ${WORK_DIR13} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME13}
# python tools/train.py ${CONFIG_FILE14} --work-dir ${WORK_DIR14} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME14}
# python tools/train.py ${CONFIG_FILE15} --work-dir ${WORK_DIR15} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME15}
# python tools/train.py ${CONFIG_FILE16} --work-dir ${WORK_DIR16} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME16}
# python tools/train.py ${CONFIG_FILE17} --work-dir ${WORK_DIR17} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME17}
# python tools/train.py ${CONFIG_FILE18} --work-dir ${WORK_DIR18} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME18}
# python tools/train.py ${CONFIG_FILE19} --work-dir ${WORK_DIR19} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME19}
# python tools/train.py ${CONFIG_FILE20} --work-dir ${WORK_DIR20} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME20}
# python tools/train.py ${CONFIG_FILE21} --work-dir ${WORK_DIR21} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME21}
# python tools/train.py ${CONFIG_FILE22} --work-dir ${WORK_DIR22} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME22}
# python tools/train.py ${CONFIG_FILE23} --work-dir ${WORK_DIR23} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME23}
python tools/train.py ${CONFIG_FILE24} --work-dir ${WORK_DIR24} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME24}
# python tools/train.py ${CONFIG_FILE25} --work-dir ${WORK_DIR25} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME25}
# python tools/train.py ${CONFIG_FILE26} --work-dir ${WORK_DIR26} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME26}
# python tools/train.py ${CONFIG_FILE27} --work-dir ${WORK_DIR27} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME27}
