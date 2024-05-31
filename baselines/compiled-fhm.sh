# python3 prompt-llama.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/mmhs/train-explanations.jsonl" 

python3 prompt-llama-zs.py \
    --annotation_filepath "/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json" \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --result_dir ../results/baselines/llama-3-zs/fhm_finegrained

python3 prompt-mistral-zs.py \
    --annotation_filepath "/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json" \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --result_dir ../results/baselines/mistral-zs/fhm_finegrained