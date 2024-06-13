LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM=/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json
FHM_CAPS=/mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/
MAMI=/mnt/data1/datasets/memes/mami/annotations/test.jsonl
MAMI_CAPS=/mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/

# Construct TF-IDF Similarity Matrix (LH, Text)
python3 bm25_wrapper.py \
    --annotation_filepath $FHM \
    --annotation_content content_text_caption \
    --caption_dir $FHM_CAPS \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_content content_text \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/textcaption2text/fhm_lh_bm25_matching.npy
    

python3 bm25_wrapper.py \
    --annotation_filepath $FHM \
    --annotation_content content_text_caption \
    --caption_dir $FHM_CAPS \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_content rationale \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/textcaption2rationale/fhm_lh_bm25_matching.npy \
    --overwrite

# For MAMI

python3 bm25_wrapper.py \
    --annotation_filepath $MAMI \
    --annotation_content content_text_caption \
    --caption_dir $MAMI_CAPS \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_content content_text \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/textcaption2text/mami_lh_bm25_matching.npy

python3 bm25_wrapper.py \
    --annotation_filepath $MAMI \
    --annotation_content content_text_caption \
    --caption_dir $MAMI_CAPS \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_content rationale \
    --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/textcaption2rationale/mami_lh_bm25_matching.npy \
    --overwrite

# python3 bm25_wrapper.py \
#     --annotation_filepath $FHM \
#     --annotation_content rationale \
#     --caption_dir $FHM_CAPS \
#     --support_filepaths $LATENT_HATRED \
#     --support_caption_dirs "" \
#     --support_content rationale \
#     --output_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/rationale2rationale/fhm_lh_bm25_matching.npy \
#     --overwrite