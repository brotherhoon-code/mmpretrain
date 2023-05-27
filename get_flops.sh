#!/bin/sh
CONFIG_NAME="ours"
CONFIG_FILE=./configs/_scratch_/IN20/${CONFIG_NAME}.py
python tools/analysis_tools/get_flops.py ${CONFIG_FILE}