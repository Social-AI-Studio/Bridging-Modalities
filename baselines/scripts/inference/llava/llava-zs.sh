
MODEL=liuhaotian/llava-v1.6-mistral-7b
EXP=zero_shot
# fhm
python3 ../../../prompt-llava-zs.py \
    --model_path liuhaotian/llava-v1.6-mistral-7b \
    --annotation_filepath /mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json \
    --caption_dir /mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/ \
    --image_dir /mnt/data1/datasets/memes/fhm/images/img/ \
    --result_dir ../../../../lh-results/baselines/$EXP/$MODEL/fhm_finegrained > ../../../lh-logs/$EXP/$MODEL/fhm.log && 
    
# mami
python3 ../../../prompt-llava-zs.py \
    --model_path liuhaotian/llava-v1.6-mistral-7b \
    --annotation_filepath /mnt/data1/datasets/memes/mami/annotations/test.jsonl \
    --caption_dir /mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/ \
    --image_dir /mnt/data1/datasets/memes/mami/images/img/test/ \
    --result_dir ../../../../lh-results/baselines/$EXP/$MODEL/mami/ >../../../lh-logs/$EXP/$MODEL/mami.log 

