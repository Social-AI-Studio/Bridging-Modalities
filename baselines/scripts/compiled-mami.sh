# python3 ../../prompt-llama.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/mmhs/train-explanations.jsonl" 

# python3 ../../prompt-mistral-zs.py \
#     --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
#     --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/blip2-opt-6.7b-coco/ \
#     --result_dir ../../../results/baselines/mistral-zs/mami

# python3 ../../prompt-llama-zs.py \
#     --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
#     --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/blip2-opt-6.7b-coco/ \
#     --result_dir ../../../results/baselines/llama-3-zs/mami

LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/truncated/explanations/train-explanations.jsonl
MMHS=/mnt/data1/datasets/temp/MMHS150K/explanations/train-explanations.jsonl

python3 ../../prompt-mistral-fs.py \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/blip2-opt-6.7b-coco/ \
    --result_dir ../../../results/baselines/mistral-fs/mami \
    --use_demonstrations \
    --demonstration_selection "random" \
    --support_filepaths $LATENT_HATRED $MMHS \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco