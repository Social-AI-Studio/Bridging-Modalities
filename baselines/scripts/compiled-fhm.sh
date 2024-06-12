# python3 ../../prompt-llama.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/mmhs/train-explanations.jsonl" 

# python3 ../../prompt-mistral-zs.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json" \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
#     --result_dir ../../../results/baselines/mistral-zs/fhm_finegrained

# python3 ../../prompt-llama-zs.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json" \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
#     --result_dir ../../../results/baselines/llama-3-zs/fhm_finegrained

LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

python3 ../../prompt-mistral-fs.py \
    --model_id mistralai/Mistral-7B-Instruct-v0.2 \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/mistral-fs/fhm_finegrained/tfidf \
    --use_demonstrations \
    --demonstration_selection "tf-idf" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/fhm_lh_tfidf_matching.npy