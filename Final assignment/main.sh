wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 4 \
    --epochs 200 \
    --lr 0.001 \
    --num-workers 9 \
    --seed 42 \
    --experiment-id "unet-training" \