
DATA_PATH="data/spirals"
N_CLASSES=2

export CUDA_VISIBLE_DEVICES=""


python -m training.experiment \
    --data-path ${DATA_PATH} \
    --n-classes ${N_CLASSES} \
    "$@"