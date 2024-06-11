LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM=/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json
FHM_CAPS=/mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/
MAMI=/mnt/data1/datasets/memes/mami/annotations/test.jsonl
MAMI_CAPS=/mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/

# Construct TF-IDF Similarity Matrix (LH, Text)
python3 tfidf_wrapper.py \
    --annotation_filepath $FHM \
    --annotation_content content_text_caption \
    --caption_dir $FHM_CAPS \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_content rationale \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/rationale/fhm_lh_tfidf_matching.npy \
    --overwrite

# Construct TF-IDF Similarity Matrix (LH, Rationale)

# Construct TF-IDF Similarity Matrix (LH + Image, Text)

# Construct TF-IDF Similarity Matrix (LH + Image, Text + Caption)

# Construct TF-IDF Similarity Matrix (LH + Image, Rationale) 