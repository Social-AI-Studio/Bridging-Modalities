LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/truncated/explanations/train-explanations.jsonl
MMHS=/mnt/data1/datasets/temp/MMHS150K/explanations/train-explanations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl
LLAMA2=meta-llama/Llama-2-7b-chat-hf
LLAMA3=meta-llama/Meta-Llama-3-8B
LLAVA=llava-hf/llava-v1.6-mistral-7b-hf

FHM_TFIDF=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_tfidf_matching.npy
FHM_BM25=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_bm25_matching.npy
FHM_CLIP=/mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_clip_Misogynistic_MEME_matching.npy
## FHM
# random

# CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-llama2-fs.py \
#     --model_id $LLAMA2 \
#     --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
#     --feature_dir "" \
#     --result_dir ../../../results/baselines/llama2-fs/fhm_finegrained/random \
#     --use_demonstrations \
#     --demonstration_selection "random" \
#     --demonstration_distribution "top-k" \
#     --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
#     --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
#     --support_feature_dirs "" "" \
#     --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices/fhm_clip_Misogynistic_MEME_matching.npy > logs/llama2-fhm-random.log &&

# #tf idf
# CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-llama2-fs.py \
#     --model_id $LLAMA2 \
#     --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
#     --feature_dir "" \
#     --result_dir ../../../results/baselines/llama2-fs/fhm_finegrained/tfidf \
#     --use_demonstrations \
#     --demonstration_selection "tf-idf" \
#     --demonstration_distribution "top-k" \
#     --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
#     --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
#     --support_feature_dirs "" "" \
#     --sim_matrix_filepath $FHM_TFIDF > logs/llama2-fhm-tfidf.log &&

# # bm 25

# CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-llama2-fs.py \
#     --model_id $LLAMA2 \
#     --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
#     --feature_dir "" \
#     --result_dir ../../../results/baselines/llama2-fs/fhm_finegrained/bm25 \
#     --use_demonstrations \
#     --demonstration_selection "bm-25" \
#     --demonstration_distribution "top-k" \
#     --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
#     --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
#     --support_feature_dirs "" "" \
#     --sim_matrix_filepath $FHM_BM25 > logs/llama2-fhm-bm25.log &&

# # clip

# CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-llama2-fs.py \
#     --model_id $LLAMA2 \
#     --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
#     --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/blip2-opt-6.7b-coco/ \
#     --feature_dir "" \
#     --result_dir ../../../results/baselines/llama2-fs/fhm_finegrained/clip \
#     --use_demonstrations \
#     --demonstration_selection "clip" \
#     --demonstration_distribution "top-k" \
#     --support_filepaths $LATENT_HATRED $MISOGYNISTIC_MEME \
#     --support_caption_dirs "" /mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco \
#     --support_feature_dirs "" "" \
#     --sim_matrix_filepath $FHM_CLIP > logs/llama2-fhm-clip.log 

# mami

