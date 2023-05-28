#!/bin/sh
CONFIG_NAME="convmixer-768-32"
CONFIG_FILE=./configs/_scratch_/IN20/${CONFIG_NAME}.py
python tools/analysis_tools/get_flops.py ${CONFIG_FILE}