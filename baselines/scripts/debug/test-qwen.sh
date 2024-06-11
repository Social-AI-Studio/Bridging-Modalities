LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MMHS=/mnt/data1/datasets/temp/MMHS150K/explanations/train-explanations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

MODEL=Qwen/Qwen2-7B-Instruct

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-qwen-fs.py \
    --model_id $MODEL \
    --debug_mode True \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/test-qwen \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/mami_lh_bm25_matching.npy


# CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-qwen-zs.py \
#     --model_id $MODEL \
#     --debug_mode True \
#     --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
#     --result_dir ../../../results/baselines/test-qwen