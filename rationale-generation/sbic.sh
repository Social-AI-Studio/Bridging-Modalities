python3 -u sbic.py \
    --annotation_filepath '/mnt/data1/datasets/hatespeech/sbicv2/originals/SBIC.v2.agg.trn.csv' \
    --output_dir "/mnt/data1/datasets/hatespeech/sbicv2/rationales/mistral-v0.3-7b" \
    --prompt_approach "few_shot_10_shots" \
    --process_label "hateful" \
    --num_splits $1 \
    --split $2
