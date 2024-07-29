MODEL=mistralai/Mistral-7B-Instruct-v0.3
EXP=zero_shot

# fhm
python3 ../../../prompt-mistral-zs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/deepfillv2/ofa-large-caption/ \
    --result_dir ../../../../lh-results/baselines/$EXP/$MODEL/fhm_finegrained  > ../../../logs/$EXP/$MODEL/fhm.log
    
# mami
python3 ../../../prompt-mistral-zs.py \
    --model_id $MODEL \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --result_dir ../../../../lh-results/baselines/$EXP/$MODEL/mami/ > ../../../logs/$EXP/$MODEL/mami.log 