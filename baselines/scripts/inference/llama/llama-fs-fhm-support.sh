# Support Set

FHM_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$1/fhm_fhm_tfidf_matching.npy
FHM_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$1/fhm_fhm_bm25_matching.npy

MAMI_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$1/mami_fhm_tfidf_matching.npy
MAMI_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$1/mami_fhm_bm25_matching.npy

FHM_ALIGNMENT_MEME=/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/annotations/alignment_rationales_all.jsonl
FHM_ALIGNMENT_CAPTIONS=/mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption
FHM_ALIGNMENT_IMG=/mnt/data1/datasets/memes/fhm/images/img/

MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

for EXP in 4 8 16
do
    EXP_NAME=${EXP}_shots
    echo $EXP_NAME
    # fhm
    # random

    python3 -u ../../../prompt-llama-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
        --caption_dir /mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/ \
        --result_dir ../../../../fhm-results/baselines/$EXP_NAME/$1/$MODEL/memes/fhm_finegrained/tfidf \
        --use_demonstrations \
        --prompt_format "single_prompt" \
        --demonstration_selection "tf-idf" \
        --demonstration_distribution "top-k" \
        --support_filepaths $FHM_ALIGNMENT_MEME \
        --support_caption_dirs $FHM_ALIGNMENT_CAPTIONS \
        --support_feature_dirs ""  \
        --sim_matrix_filepath $FHM_TFIDF \
        --shots $EXP > ../../../fhm-logs/$EXP_NAME/$1/$MODEL/fhm-tfidf.log

    python3 -u ../../../prompt-llama-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
        --caption_dir /mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/ \
        --result_dir ../../../../fhm-results/baselines/$EXP_NAME/$1/$MODEL/memes/fhm_finegrained/bm25 \
        --use_demonstrations \
        --prompt_format "single_prompt" \
        --demonstration_selection "bm-25" \
        --demonstration_distribution "top-k" \
        --support_filepaths $FHM_ALIGNMENT_MEME \
        --support_caption_dirs $FHM_ALIGNMENT_CAPTIONS \
        --support_feature_dirs ""  \
        --sim_matrix_filepath $FHM_BM25 \
        --shots $EXP > ../../../fhm-logs/$EXP_NAME/$1/$MODEL/fhm-bm25.log

    # # mistral-mami
    # # # random

    python3 -u ../../../prompt-llama-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
        --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
        --result_dir ../../../../fhm-results/baselines/$EXP_NAME/$1/$MODEL/mami/tfidf \
        --use_demonstrations \
        --prompt_format "single_prompt" \
        --demonstration_selection "tf-idf" \
        --demonstration_distribution "top-k" \
        --support_filepaths $FHM_ALIGNMENT_MEME \
        --support_caption_dirs $FHM_ALIGNMENT_CAPTIONS \
        --support_feature_dirs ""  \
        --sim_matrix_filepath $MAMI_TFIDF \
        --shots $EXP > ../../../fhm-logs/$EXP_NAME/$1/$MODEL/mami-tfidf.log

    python3 -u ../../../prompt-llama-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
        --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
        --result_dir ../../../../fhm-results/baselines/$EXP_NAME/$1/$MODEL/mami/bm25 \
        --use_demonstrations \
        --prompt_format "single_prompt" \
        --demonstration_selection "bm-25" \
        --demonstration_distribution "top-k" \
        --support_filepaths $FHM_ALIGNMENT_MEME \
        --support_caption_dirs $FHM_ALIGNMENT_CAPTIONS \
        --support_feature_dirs ""  \
        --sim_matrix_filepath $MAMI_BM25 \
        --shots $EXP > ../../../fhm-logs/$EXP_NAME/$1/$MODEL/mami-bm25.log

done