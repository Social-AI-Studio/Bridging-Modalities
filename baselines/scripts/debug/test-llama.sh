LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/truncated/explanations/train-explanations.jsonl
MMHS=/mnt/data1/datasets/temp/MMHS150K/explanations/train-explanations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl
LLAMA2=meta-llama/Llama-2-7b-hf
LLAMA3=meta-llama/Meta-Llama-3-8B

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-llama2-fs.py \
    --model_id meta-llama/Llama-2-7b-chat-hf \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/test-llama \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_bm25_matching.npy 