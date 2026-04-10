#!/bin/bash

# 1. 基础参数设置
dataname='BRATS2023'
datapath='/media/s2/wyb/Dataset/BraTS2023PRE'   # 必须改成你实际预处理后npy的绝对路径
savepath='output_brats2023'        # 训练结果保存的文件夹名
 
# 2. 显卡设置 (假设你只用1张卡，所以填 0)
export CUDA_VISIBLE_DEVICES=0,2

# 3. 指定 Python
# 因为你在外面会先 conda activate 你的环境，所以这里直接写 python 即可，它会自动调用你 conda 里的 python。
PYTHON=python

# 4. 启动训练 (注意：如果显存不够报 OOM 错误，把 --batch_size=2 改成 1)
$PYTHON train.py \
    --batch_size=2 \
    --iter_per_epoch 150 \
    --datapath $datapath \
    --savepath $savepath \
    --num_epochs 300 \
    --lr 2e-4 \
    --region_fusion_start_epoch 20 \
    --dataname $dataname