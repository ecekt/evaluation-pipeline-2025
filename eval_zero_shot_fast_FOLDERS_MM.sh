#!/bin/bash

MODEL_FOLDER="/data/storage/data_babylm/llava_baby_IMGPOOLED_trained_checkpoints_mm_first/babyllava_"
REVISION_NAME=$2
BACKEND=$3
EVAL_DIR=${4:-"evaluation_data/fast_eval"}

if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi


CHECKPOINTS=(1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000 200000000 300000000 400000000 500000000 600000000 700000000 800000000 900000000 lastepoch_mm_first)

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    MODEL_PATH="${MODEL_FOLDER}${CHECKPOINT}"
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/blimp_fast" --save_predictions --revision_name $REVISION_NAME
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/supplement_fast" --save_predictions --revision_name $REVISION_NAME
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EVAL_DIR}/ewok_fast" --save_predictions --revision_name $REVISION_NAME
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task wug_adj --data_path "${EVAL_DIR}/wug_adj_nominalization" --save_predictions --revision_name $REVISION_NAME
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task wug_past --data_path "${EVAL_DIR}/wug_past_tense" --save_predictions --revision_name $REVISION_NAME
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking_fast" --save_predictions --revision_name $REVISION_NAME
    #python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND_READ --data_path "${EVAL_DIR}/reading/reading_data.csv" --revision_name $REVISION_NAME
done