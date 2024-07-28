MODEL=meta-llama/Meta-Llama-3-8B-Instruct

LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl

FHM_RANDOM=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/500_8500_matching.npy
MAMI_RANDOM=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/1000_8500_matching.npy

FHM_ALIGNMENT_MEME=/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/annotations/alignment_rationales_all.jsonl
FHM_ALIGNMENT_CAPTIONS=/mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption


# CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-llama-zs.py \
#     --model_id $MODEL \
#     --debug_mode True \
#     --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
#     --result_dir ../../../results/baselines/test-llama 

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-llama-fs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../test-fhm-results/baselines/$EXP_NAME/random/$MODEL/fhm_finegrained/random \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $FHM_ALIGNMENT_MEME \
    --support_caption_dirs $FHM_ALIGNMENT_CAPTIONS \
    --support_feature_dirs ""  \
    --sim_matrix_filepath $FHM_RANDOM \
    --shots 4 \
    
