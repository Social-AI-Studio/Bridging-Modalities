# python3 prompt-llama.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/mmhs/train-explanations.jsonl" 

# python3 prompt-mistral-zs.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json" \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
#     --result_dir ../results/baselines/mistral-zs/fhm_finegrained

# python3 prompt-llama-zs.py \
#     --annotation_filepath "/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json" \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
#     --result_dir ../results/baselines/llama-3-zs/fhm_finegrained

LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/truncated/explanations/train-explanations.jsonl
MMHS=/mnt/data1/datasets/temp/MMHS150K/explanations/train-explanations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

python3 prompt-mistral-fs-no-rationales.py \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
    --feature_dir "" \
    --result_dir ../results/baselines/mistral-fs-no-rationales/fhm_finegrained \
    --use_demonstrations \
    --demonstration_selection "clip" \
    --demonstration_distribution "equal" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/memes/Misogynistic_MEME/preprocessing/blip2_captions/combined \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_clip_Misogynistic_MEME_matching.npy