LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

# Construct TF-IDF Similarity Matrix (LH, Text)
python3 tfidf_wrapper.py \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --annotation_content content_text \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_content content_text \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/fhm_lh_tfidf_matching.npy \
    --overwrite

# Construct TF-IDF Similarity Matrix (LH, Rationale)

# Construct TF-IDF Similarity Matrix (LH + Image, Text)

# Construct TF-IDF Similarity Matrix (LH + Image, Text + Caption)

# Construct TF-IDF Similarity Matrix (LH + Image, Rationale) 