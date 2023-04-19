#!/bin/sh

# CONFIG_NAME1=resnet50_8xb32_in1k
CONFIG_NAME1=config_carrot-in10
CONFIG_NAME2=config_resnet-in50
CONFIG_NAME3=config_seresnet-in10
CONFIG_NAME3_2=config_seresnet-in10-add
CONFIG_NAME3_3=config_seresnet-in10-tanh
CONFIG_NAME4=config_cbam-in10

CONFIG_NAME5=config_nonlocalnet-in10-s3-concat
CONFIG_NAME6=config_nonlocalnet-in10-s3-embedgau
CONFIG_NAME7=config_nonlocalnet-in10-s3-dotpro
CONFIG_NAME8=config_nonlocalnet-in10-s3-gau

CONFIG_NAME9=config_nonlocalnet2-in10-s3-concat
CONFIG_NAME10=config_nonlocalnet2-in10-s3-embedgau
CONFIG_NAME11=config_nonlocalnet2-in10-s3-dotpro
CONFIG_NAME12=config_nonlocalnet2-in10-s3-gau

CONFIG_NAME13=config_gcnet-in10
CONFIG_NAME14=config_gcnet2-in10


# /home/brotherhoon88/workspace/mmpretrain/configs/resnet/resnet50_8xb32_in1k.py

CONFIG_FILE1=./configs/_scratch_/${CONFIG_NAME1}.py
CONFIG_FILE2=./configs/_scratch_/${CONFIG_NAME2}.py
CONFIG_FILE3=./configs/_scratch_/${CONFIG_NAME3}.py 
CONFIG_FILE3_2=./configs/_scratch_/${CONFIG_NAME3_2}.py 
CONFIG_FILE3_3=./configs/_scratch_/${CONFIG_NAME3_3}.py 
CONFIG_FILE4=./configs/_scratch_/${CONFIG_NAME4}.py
CONFIG_FILE5=./configs/_scratch_/${CONFIG_NAME5}.py
CONFIG_FILE6=./configs/_scratch_/${CONFIG_NAME6}.py
CONFIG_FILE7=./configs/_scratch_/${CONFIG_NAME7}.py
CONFIG_FILE8=./configs/_scratch_/${CONFIG_NAME8}.py
CONFIG_FILE9=./configs/_scratch_/${CONFIG_NAME9}.py
CONFIG_FILE10=./configs/_scratch_/${CONFIG_NAME10}.py
CONFIG_FILE11=./configs/_scratch_/${CONFIG_NAME11}.py
CONFIG_FILE12=./configs/_scratch_/${CONFIG_NAME12}.py
CONFIG_FILE13=./configs/_scratch_/${CONFIG_NAME13}.py
CONFIG_FILE14=./configs/_scratch_/${CONFIG_NAME14}.py


WORK_DIR1=./work_dir/${CONFIG_NAME1}
WORK_DIR2=./work_dir/${CONFIG_NAME2}
WORK_DIR3=./work_dir/${CONFIG_NAME3}
WORK_DIR3_2=./work_dir/${CONFIG_NAME3_2}
WORK_DIR3_3=./work_dir/${CONFIG_NAME3_3}
WORK_DIR4=./work_dir/${CONFIG_NAME4}
WORK_DIR5=./work_dir/${CONFIG_NAME5}
WORK_DIR6=./work_dir/${CONFIG_NAME6}
WORK_DIR7=./work_dir/${CONFIG_NAME7}
WORK_DIR8=./work_dir/${CONFIG_NAME8}
WORK_DIR9=./work_dir/${CONFIG_NAME9}
WORK_DIR10=./work_dir/${CONFIG_NAME10}
WORK_DIR11=./work_dir/${CONFIG_NAME11}
WORK_DIR12=./work_dir/${CONFIG_NAME12}
WORK_DIR13=./work_dir/${CONFIG_NAME13}
WORK_DIR14=./work_dir/${CONFIG_NAME14}


EXP_NAME1=${CONFIG_NAME1}-sgd-b32
EXP_NAME2=${CONFIG_NAME2}-sgd-b32
EXP_NAME3=${CONFIG_NAME3}-sgd-b32
EXP_NAME3_2=${CONFIG_NAME3_2}-sgd-b32
EXP_NAME3_3=${CONFIG_NAME3_3}-sgd-b32
EXP_NAME4=${CONFIG_NAME4}-sgd-b32
EXP_NAME5=${CONFIG_NAME5}-sgd-b32
EXP_NAME6=${CONFIG_NAME6}-sgd-b32
EXP_NAME7=${CONFIG_NAME7}-sgd-b32
EXP_NAME8=${CONFIG_NAME8}-sgd-b32
EXP_NAME9=${CONFIG_NAME9}-sgd-b32
EXP_NAME10=${CONFIG_NAME10}-sgd-b32
EXP_NAME11=${CONFIG_NAME11}-sgd-b32
EXP_NAME12=${CONFIG_NAME12}-sgd-b32
EXP_NAME13=${CONFIG_NAME13}-sgd-b32
EXP_NAME14=${CONFIG_NAME14}-sgd-b32


SEED=42

# python tools/train.py ${CONFIG_FILE1} --work-dir ${WORK_DIR1} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME1}
# python tools/train.py ${CONFIG_FILE2} --work-dir ${WORK_DIR2} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME2}
# python tools/train.py ${CONFIG_FILE3} --work-dir ${WORK_DIR3} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME3}
# python tools/train.py ${CONFIG_FILE3_2} --work-dir ${WORK_DIR3_2} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME3_2}
python tools/train.py ${CONFIG_FILE3_3} --work-dir ${WORK_DIR3_3} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME3_3}
# python tools/train.py ${CONFIG_FILE4} --work-dir ${WORK_DIR4} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME4}

# python tools/train.py ${CONFIG_FILE5} --work-dir ${WORK_DIR5} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME5}
# python tools/train.py ${CONFIG_FILE6} --work-dir ${WORK_DIR6} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME6}
# python tools/train.py ${CONFIG_FILE7} --work-dir ${WORK_DIR7} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME7}
# python tools/train.py ${CONFIG_FILE8} --work-dir ${WORK_DIR8} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME8}

# python tools/train.py ${CONFIG_FILE9} --work-dir ${WORK_DIR9} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME9}
# python tools/train.py ${CONFIG_FILE10} --work-dir ${WORK_DIR10} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME10}
# python tools/train.py ${CONFIG_FILE11} --work-dir ${WORK_DIR11} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME11}
# python tools/train.py ${CONFIG_FILE12} --work-dir ${WORK_DIR12} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME12}

# python tools/train.py ${CONFIG_FILE13} --work-dir ${WORK_DIR13} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME13}
# python tools/train.py ${CONFIG_FILE14} --work-dir ${WORK_DIR14} --cfg-options randomness.seed=${SEED} visualizer.vis_backends.0.init_kwargs.name=${EXP_NAME14}
