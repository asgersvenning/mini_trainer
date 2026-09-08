DIR="/mnt/c/Users/asger/Downloads/gefion_experiment_metrics_25082026"
SIZE="vit_l_16"

DATASET="plantnet"
EVAL_DATASET="plantnet"
COMB_NAME="${DATASET}"
if [[ $DATASET != $EVAL_DATASET ]]; then
    COMB_NAME="${DATASET}_${EVAL_DATASET}"
fi

HEADS=("flat" "hierarchical" "conditional" "independent")
for head in "${HEADS[@]}"; do
    FILE="${DIR}/${SIZE}/results/runs/${head}_${DATASET}/predict/${EVAL_DATASET}/mini_metric.csv"
    DST_DIR="${DIR}/updated_boot_metrics/${EVAL_DATASET}/${SIZE}"
    mkdir -p $DST_DIR
    DST="${DST_DIR}/${head}.csv"
    COMBINATIONS="${DIR}/${SIZE}/results/combinations/${COMB_NAME}.csv"
    uv run boot_metrics.py --input "$FILE" --output "$DST" --combinations "$COMBINATIONS" -n 100 --seed 123
done