LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

python3 sift_wrapper.py \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --feature_dir /mnt/data1/datasets/memes/cmtl-rag/fhm/embeddings/sift \
    --image_dir /mnt/data1/datasets/memes/fhm/images/img \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/memes/Misogynistic_MEME/preprocessing/blip2_captions/combined \
    --support_feature_dirs "" /mnt/data1/datasets/memes/cmtl-rag/Misogynistic_MEME/embeddings/sift \
    --support_image_dirs "" /mnt/data1/datasets/memes/Misogynistic_MEME/images/img_clean/combined \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_sift_Misogynistic_MEME_matching.npy