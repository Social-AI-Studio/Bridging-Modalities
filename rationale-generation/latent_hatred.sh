python3 latent_hatred_single_prompt.py \
    --annotation_filepath '/mnt/data1/datasets/hatespeech/latent_hatred/annotations/sampled_demonstrations_nonhate.jsonl' \
    --output_dir "/mnt/data1/datasets/hatespeech/latent_hatred/rationales/mistral-v0.3-7b" \
    --prompt_approach "few_shot" \
    --num_splits 1 \
    --split 0
