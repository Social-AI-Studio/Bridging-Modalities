LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/truncated/explanations/train-explanations.jsonl
MMHS=/mnt/data1/datasets/temp/MMHS150K/explanations/train-explanations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

python3 clip_wrapper.py \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
    --feature_dir /mnt/data1/datasets/memes/cmtl-rag/fhm/embeddings/clip-ViT-B-32 \
    --support_filepaths $LATENT_HATRED $MMHS \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" /mnt/data1/datasets/memes/cmtl-rag/mmhs/embeddings/clip-ViT-B-32 \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_clip_mmhs_matching.npy