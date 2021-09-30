#!/bin/bash

cd /ccir/data/code/predict
python run_trained_JointBert.py
python run_trained_interact1.py
python run_trained_interact3.py
python integration.py
echo '模型已全部推理完成，结果result.json已保存在prediction_result文件夹下'
