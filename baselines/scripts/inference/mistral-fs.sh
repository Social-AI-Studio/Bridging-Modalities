LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/fhm_lh_tfidf_matching.npy
FHM_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/fhm_lh_bm25_matching.npy

MAMI_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/mami_lh_tfidf_matching.npy
MAMI_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/text/mami_lh_bm25_matching.npy

MODEL=mistralai/Mistral-7B-Instruct-v0.3
EXP=four_shot
#fhm
# random

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained/random \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath $FHM_TFIDF \
    --shots 4 >../../logs/$EXP/$MODEL/fhm-random.log &&

#tf idf
CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained/tfidf \
    --use_demonstrations \
    --demonstration_selection "tf-idf" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath $FHM_TFIDF \
    --shots 4 >../../logs/$EXP/$MODEL/fhm-tfidf.log &&

# bm 25

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained/bm25 \
    --use_demonstrations \
    --demonstration_selection "bm-25" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath $FHM_BM25 \
    --shots 4 >../../logs/$EXP/$MODEL/fhm-bm25.log &&

# mistral-mami
# random

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/random \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath $MAMI_TFIDF \
    --shots 4 >../../logs/$EXP/$MODEL/mami-random.log &&


# tf-idf

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/tfidf \
    --use_demonstrations \
    --demonstration_selection "tf-idf" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath $MAMI_TFIDF \
    --shots 4 >../../logs/$EXP/$MODEL/mami-tfidf.log &&

# bm-25

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/bm25 \
    --use_demonstrations \
    --demonstration_selection "bm-25" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs "" \
    --sim_matrix_filepath $MAMI_BM25 \
    --shots 4 >../../logs/$EXP/$MODEL/mami-bm25.log 
