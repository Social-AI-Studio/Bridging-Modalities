LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl

FHM_RANDOM=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/500_21472_matching.npy
MAMI_RANDOM=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/1000_21472_matching.npy

FHM_ALIGNMENT_MEME=/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/annotations/alignment_rationales_all.jsonl
FHM_ALIGNMENT_CAPTIONS=/mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption
FHM_ALIGNMENT_IMG=/mnt/data1/datasets/memes/fhm/images/img/

MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

for EXP in 16
do
    EXP_NAME=${EXP}_shots
    echo $EXP_NAME
    # fhm
    # random

    # python3 -u ../../../prompt-llama-fs.py \
    #     --model_id $MODEL \
    #     --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    #     --caption_dir /mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/ \
    #     --result_dir ../../../../fhm-results/baselines/$EXP_NAME/random/$MODEL/memes/fhm_finegrained/random \
    #     --use_demonstrations \
    #     --prompt_format "single_prompt" \
    #     --demonstration_selection "random" \
    #     --demonstration_distribution "top-k" \
    #     --support_filepaths $FHM_ALIGNMENT_MEME \
    #     --support_caption_dirs $FHM_ALIGNMENT_CAPTIONS \
    #     --support_feature_dirs ""  \
    #     --sim_matrix_filepath $FHM_RANDOM \
    #     --shots $EXP > ../../../fhm-logs/$EXP_NAME/random/$MODEL/fhm-random.log

    # # # mistral-mami
    # # # # random

    python3 -u ../../../prompt-llama-fs.py \
        --model_id $MODEL \
        --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
        --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
        --result_dir ../../../../fhm-results/baselines/$EXP_NAME/random/$MODEL/mami/random \
        --use_demonstrations \
        --prompt_format "single_prompt" \
        --demonstration_selection "random" \
        --demonstration_distribution "top-k" \
        --support_filepaths $FHM_ALIGNMENT_MEME \
        --support_caption_dirs $FHM_ALIGNMENT_CAPTIONS \
        --support_feature_dirs ""  \
        --sim_matrix_filepath $MAMI_RANDOM \
        --shots $EXP > ../../../fhm-logs/$EXP_NAME/random/$MODEL/mami-random.log

done