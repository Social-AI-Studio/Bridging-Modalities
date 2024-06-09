LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/truncated/explanations/train-explanations.jsonl
MMHS=/mnt/data1/datasets/temp/MMHS150K/explanations/train-explanations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_tfidf_matching.npy
FHM_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_bm25_matching.npy
FHM_CLIP=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_clip_Misogynistic_MEME_matching.npy

MAMI_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_tfidf_matching.npy
MAMI_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_bm25_matching.npy
MAMI_CLIP=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_clip_Misogynistic_MEME_matching.npy

MODEL=mistralai/Mistral-7B-Instruct-v0.3
EXP=few_shot
#fhm
# random

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained/random \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_clip_Misogynistic_MEME_matching.npy >../../logs/$EXP/$MODEL/fhm-random.log &&

#tf idf
CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained/tfidf \
    --use_demonstrations \
    --demonstration_selection "tf-idf" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_tfidf_matching.npy >../../logs/$EXP/$MODEL/fhm-tfidf.log &&

# bm 25

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained/bm25 \
    --use_demonstrations \
    --demonstration_selection "bm-25" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_bm25_matching.npy >../../logs/$EXP/$MODEL/fhm-bm25.log &&

# clip

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained/clip \
    --use_demonstrations \
    --demonstration_selection "clip" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_clip_Misogynistic_MEME_matching.npy >../../logs/$EXP/$MODEL/fhm-clip.log &&

# mistral-mami
# random

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/blip2-opt-6.7b-coco \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/random \
    --use_demonstrations \
    --demonstration_selection "random" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_tfidf_matching.npy >../../logs/$EXP/$MODEL/mami-random.log &&


# tf-idf

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/blip2-opt-6.7b-coco \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/tfidf \
    --use_demonstrations \
    --demonstration_selection "tf-idf" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_tfidf_matching.npy >../../logs/$EXP/$MODEL/mami-tfidf.log &&

# bm-25

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/blip2-opt-6.7b-coco \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/bm25 \
    --use_demonstrations \
    --demonstration_selection "bm-25" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_bm25_matching.npy >../../logs/$EXP/$MODEL/mami-bm25.log &&


## clip

CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-mistral-fs-no-rationales.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/blip2-opt-6.7b-coco \
    --feature_dir "" \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/clip \
    --use_demonstrations \
    --demonstration_selection "clip" \
    --demonstration_distribution "top-k" \
    --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
    --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
    --support_feature_dirs "" "" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/mami_clip_Misogynistic_MEME_matching.npy >../../logs/$EXP/$MODEL/mami-clip.log 