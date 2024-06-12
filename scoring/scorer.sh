LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM=/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json
FHM_CAPS=/mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/
MAMI=/mnt/data1/datasets/memes/mami/annotations/test.jsonl
MAMI_CAPS=/mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/

python3 retrieval_scorer.py \
    --annotation_filepath $MAMI \
    --caption_dir $MAMI_CAPS \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/mami_lh_bm25_matching.npy \
    --top_p 1