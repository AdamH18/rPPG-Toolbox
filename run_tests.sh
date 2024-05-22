#!/bin/sh

cd /home/adamh/rPPG-Toolbox
conda activate rppg-toolbox
python main.py --config_file $1
