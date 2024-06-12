import os
import json

ANNOTATIONS_FILEPATH = "/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/annotations/alignment.jsonl"
RATIONALE_DIR = "/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/rationales/mistral-v0.3-7b/few_shot_10_shots_cleaned"
OUTPUT_FILEPATH = "/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/annotations/alignment_rationales_{k}.jsonl"

def load_rationale(content_id, rationales_dir):
    caption_filepath = os.path.join(rationales_dir, f"{content_id}.json")
    with open(caption_filepath) as f:
        d = json.load(f)

    return d['rationale']

def load_annotations(annotation_filepath):
    annotations = []
    with open(annotation_filepath) as f:
        for line in f:
            annotations.append(json.loads(line))

    return annotations

annotations = load_annotations(ANNOTATIONS_FILEPATH)


target_mapping = {
    'pc_empty': 0, 
    'sex': 1, 
    'race': 2, 
    'religion': 3,
    'nationality': 4, 
    'disability': 5, 
}

targets = []
for annot in annotations:
    annot["class_binarized"] = 1 if annot['gold_hate'][0] == "hateful" else 0
    annot["target_categories_mapped"] = [target_mapping[x] for x in annot['gold_pc']]
    annot["mistral_instruct_statement"] = load_rationale(f"{annot['id']:05}", RATIONALE_DIR)

pc_records = {
    'nationality': [], 
    'pc_empty': [], 
    'disability': [], 
    'religion': [], 
    'race': [], 
    'sex': []
}
for annot in annotations:
    pc_records[annot['gold_pc'][0]].append(annot)

import random 
for k in [4,8,16,32]:
    records = []
    for v in pc_records.values():
        records += random.sample(v, k)

    print(f"Num. Records (k={k}):", len(records))

    output_filepath = OUTPUT_FILEPATH.format(k=k)
    with open(output_filepath, "w+") as f:
        for obj in records:
            json.dump(obj, f)
            f.write("\n")