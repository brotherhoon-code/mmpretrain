#!/bin/sh
CONFIG_FILE1=./configs/__scratch/config_custom_resnet.py # resnet
CONFIG_FILE2=./configs/resnet/resnet50_8xb32_in1k.py

python tools/analysis_tools/get_flops.py ${CONFIG_FILE1} --shape 32 32
# python tools/analysis_tools/get_flops.py ${CONFIG_FILE2}