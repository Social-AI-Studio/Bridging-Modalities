python3 -u latent_hatred_single_prompt.py \
    --annotation_filepath '/mnt/data1/datasets/hatespeech/latent_hatred/annotations/annotations.jsonl' \
    --output_dir "/mnt/data1/datasets/hatespeech/latent_hatred/rationales/mistral-v0.3-7b" \
    --prompt_approach "few_shot_10_shots" \
    --num_splits $1 \
    --split $2
