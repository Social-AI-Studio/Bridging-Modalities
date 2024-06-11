MODEL=gemini-1.5-pro
EXP=zero_shot
# fhm
CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-geminipro-zs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --result_dir ../../../results/baselines/$EXP/$MODEL/fhm_finegrained  >../../logs/$EXP/$MODEL/fhm.log && 
    
# mami
CUDA_VISIBLE_DEVICES=0 python3 ../../prompt-geminipro-zs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --result_dir ../../../results/baselines/$EXP/$MODEL/mami/ >../../logs/$EXP/$MODEL/mami.log 
