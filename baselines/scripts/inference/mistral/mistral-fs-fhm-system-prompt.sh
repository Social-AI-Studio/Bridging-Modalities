LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl

FHM_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$1/fhm_lh_tfidf_matching.npy
FHM_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized//$1/fhm_lh_bm25_matching.npy

MAMI_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$1/mami_lh_tfidf_matching.npy
MAMI_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$1/mami_lh_bm25_matching.npy

MODEL=mistralai/Mistral-7B-Instruct-v0.3

ln -s /mnt/data1/CMTL-RAG/utils.py /mnt/data1/CMTL-RAG/baselines/
ln -s /mnt/data1/CMTL-RAG/matching /mnt/data1/CMTL-RAG/baselines/

for EXP in 4
do
    EXP_NAME=${EXP}_shots
    echo $EXP_NAME

    #tf idf
    python3 -u ../../prompt-mistral-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
        --caption_dir /mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/multi_turn_prompt/$EXP_NAME/$1/$MODEL/fhm_finegrained/tfidf \
        --prompt_format "multi_turn_prompt" \
        --use_demonstrations \
        --demonstration_selection "tf-idf" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs "" \
        --sim_matrix_filepath $FHM_TFIDF \
        --shots $EXP # > ../../logs/multi_turn_prompt/$EXP_NAME/$1/$MODEL/fhm-tfidf.log

done