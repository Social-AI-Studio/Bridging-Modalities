python3 alignment.py \
    --annotation_filepath "/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/annotations/alignment.jsonl" \
    --img_dir "/mnt/data1/datasets/memes/fhm/images/deepfillv2" \
    --captions_dir "/mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption" \
    --web_entities_dir "/mnt/data1/datasets/memes/fhm/web_entities" \
    --output_dir "/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/rationales/mistral-v0.3-7b" \
    --prompt_approach "few_shot_10_shots" \
    --num_splits $1 \
    --split $2
