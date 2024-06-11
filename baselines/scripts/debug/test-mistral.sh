LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

MODEL=mistralai/Mistral-7B-Instruct-v0.3

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-zs.py \
    --model_id $MODEL \
    --debug_mode True \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --result_dir ../../../results/baselines/test-mistral &&

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs.py \
    --model_id $MODEL \
    --debug_mode True \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/test-mistral \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/rationale/fhm_lh_bm25_matching.npy \
    --shots 8