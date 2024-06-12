LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

python3 clip_wrapper.py \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --feature_dir /mnt/data1/datasets/memes/cmtl-rag/mami/embeddings/clip-ViT-B-32 \
    --image_dir /mnt/data1/datasets/memes/mami/images/deepfillv2/test \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/memes/Misogynistic_MEME/preprocessing/blip2_captions/combined \
    --support_feature_dirs "" /mnt/data1/datasets/memes/cmtl-rag/Misogynistic_MEME/embeddings/clip-ViT-B-32 \
    --support_image_dirs "" /mnt/data1/datasets/memes/Misogynistic_MEME/images/img_clean/combined \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_clip_Misogynistic_MEME_matching.npy