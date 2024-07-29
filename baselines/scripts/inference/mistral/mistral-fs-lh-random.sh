LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl

FHM_RANDOM=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/500_21472_matching.npy
MAMI_RANDOM=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/1000_21472_matching.npy

MODEL=mistralai/Mistral-7B-Instruct-v0.3

for EXP in 4 8 16
do
    EXP_NAME=${EXP}_shots
    echo $EXP_NAME
    # fhm
    # random

    python3 -u ../../prompt-mistral-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
        --caption_dir /mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/random/$MODEL/memes/fhm_finegrained/random \
        --use_demonstrations \
        --demonstration_selection "random" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs "" \
        --sim_matrix_filepath $FHM_RANDOM \
        --shots $EXP > ../../logs/$EXP_NAME/random/$MODEL/fhm-random.log

    # mistral-mami
    # # random

    python3 -u ../../prompt-mistral-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
        --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
        --feature_dir "" \
        --result_dir ../../../results/baselines/$EXP_NAME/random/$MODEL/mami/random \
        --use_demonstrations \
        --demonstration_selection "random" \
        --demonstration_distribution "top-k" \
        --support_filepaths $LATENT_HATRED \
        --support_caption_dirs "" \
        --support_feature_dirs "" \
        --sim_matrix_filepath $MAMI_RANDOM \
        --shots $EXP > ../../logs/$EXP_NAME/random/$MODEL/mami-random.log

done