LATENT_HATRED=/mnt/sda/mshee/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM_TFIDF=/mnt/sda/mshee/datasets/cmtl-rag/sim_matrices_finalized/$1/fhm_lh_tfidf_matching.npy
FHM_BM25=/mnt/sda/mshee/datasets/cmtl-rag/sim_matrices_finalized/$1/fhm_lh_bm25_matching.npy

MAMI_TFIDF=/mnt/sda/mshee/datasets/cmtl-rag/sim_matrices_finalized/$1/mami_lh_tfidf_matching.npy
MAMI_BM25=/mnt/sda/mshee/datasets/cmtl-rag/sim_matrices_finalized/$1/mami_lh_bm25_matching.npy

MODEL=Qwen/Qwen2-7B-Instruct

for EXP in 4 8 16
do
    EXP_NAME=${EXP}_shots
    echo $EXP_NAME
    #fhm
    # random

    python3 ../../prompt-qwen-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/sda/mshee/datasets/fhm_finegrained/annotations/dev_seen.json \
        --caption_dir /mnt/sda/mshee/datasets/fhm/captions/deepfillv2/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/$1/$MODEL/fhm_finegrained/random \
        --use_demonstrations \
        --demonstration_selection "random" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs ""  \
        --sim_matrix_filepath $FHM_BM25 \
        --shots $EXP > ../../logs/$EXP_NAME/$1/$MODEL/fhm-random.log 

    #tf idf
    python3 ../../prompt-qwen-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/sda/mshee/datasets/fhm_finegrained/annotations/dev_seen.json \
        --caption_dir /mnt/sda/mshee/datasets/fhm/captions/deepfillv2/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/$1/$MODEL/fhm_finegrained/tfidf \
        --use_demonstrations \
        --demonstration_selection "tf-idf" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED  \
        --support_caption_dirs "" \
        --support_feature_dirs ""  \
        --sim_matrix_filepath $FHM_TFIDF \
        --shots $EXP > ../../logs/$EXP_NAME/$1/$MODEL/fhm-tfidf.log 

    # bm 25

    python3 ../../prompt-qwen-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/sda/mshee/datasets/fhm_finegrained/annotations/dev_seen.json \
        --caption_dir /mnt/sda/mshee/datasets/fhm/captions/deepfillv2/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/$1/$MODEL/fhm_finegrained/bm25 \
        --use_demonstrations \
        --demonstration_selection "bm-25" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs "" \
        --sim_matrix_filepath $FHM_BM25 \
        --shots $EXP > ../../logs/$EXP_NAME/$1/$MODEL/fhm-bm25.log 


    # qwen-mami
    # random

    python3 ../../prompt-qwen-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/sda/mshee/datasets/mami/annotations/test.jsonl \
        --caption_dir /mnt/sda/mshee/datasets/mami/captions/deepfillv2/test/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/$1/$MODEL/mami/random \
        --use_demonstrations \
        --demonstration_selection "random" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs "" \
        --sim_matrix_filepath $FHM_TFIDF \
        --shots $EXP >../../logs/$EXP_NAME/$1/$MODEL/mami-random.log 


    # tf-idf

    python3 ../../prompt-qwen-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/sda/mshee/datasets/mami/annotations/test.jsonl \
        --caption_dir /mnt/sda/mshee/datasets/mami/captions/deepfillv2/test/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/$1/$MODEL/mami/tfidf \
        --use_demonstrations \
        --demonstration_selection "tf-idf" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs "" \
        --sim_matrix_filepath $MAMI_TFIDF \
        --shots $EXP >../../logs/$EXP_NAME/$1/$MODEL/mami-tfidf.log 

    # bm-25

    python3 ../../prompt-qwen-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/sda/mshee/datasets/mami/annotations/test.jsonl \
        --caption_dir /mnt/sda/mshee/datasets/mami/captions/deepfillv2/test/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/$1/$MODEL/mami/bm25 \
        --use_demonstrations \
        --demonstration_selection "bm-25" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs "" \
        --sim_matrix_filepath $MAMI_BM25 \
        --shots $EXP >../../logs/$EXP_NAME/$1/$MODEL/mami-bm25.log
done