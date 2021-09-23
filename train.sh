python3 train.py \
    models/pathology/r101_dh256.py \
    --gpus=1 \
    --workers=4 \
    --exp-name=pathology \
    --batch-size=10