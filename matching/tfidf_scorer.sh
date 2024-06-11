LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

# Construct TF-IDF Similarity Matrix (LH, Text)
python3 tfidf_retrieval_scorer.py \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/rationale/fhm_lh_tfidf_matching.npy \
    --top_p 1