#!/bin/sh
CONFIG_NAME="resnet50"
CONFIG_FILE=./configs/_scratch_/IN40/${CONFIG_NAME}.py
python tools/analysis_tools/get_flops.py ${CONFIG_FILE}