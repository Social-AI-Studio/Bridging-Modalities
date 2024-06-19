python3 -u mami.py \
    --annotation_filepath "/mnt/data1/datasets/memes/mami/annotations/test.jsonl" \
    --img_dir "/mnt/data1/datasets/memes/mami/images/test/deepfillv2" \
    --captions_dir "/mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption" \
    --web_entities_dir "/mnt/data1/datasets/memes/mami/entities/deepfillv2/test/" \
    --output_dir "/mnt/data1/datasets/memes/mami/projects/CMTL-RAG/rationales/mistral-v0.3-7b" \
    --prompt_approach "few_shot_10_shots" \
    --num_splits $1 \
    --split $2
