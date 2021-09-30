#!/bin/bash

echo 'start train the first model--JointBert'
cd /ccir/data/code/scripts
python train_jointBert.py

echo 'start train the second model--InteractModel1'
python train_interact1.py

echo 'start train the third model--InteractModel3'
python train_interact3.py

cd ../predict
python run_JointBert.py
python run_interact1.py 
python run_interact3.py 
python integration.py
echo '模型已全部推理完成，结果result.json已保存在prediction_result文件夹下'
