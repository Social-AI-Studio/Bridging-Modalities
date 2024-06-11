import os
import json

ANNOTATIONS_FILEPATH = "/mnt/data1/datasets/hatespeech/latent_hatred/annotations/annotations.jsonl"
RATIONALE_DIR = "/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/rationales/mistral-v0.3-7b/few_shot_10_shots_cleaned"
OUTPUT_FILEPATH = "/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl"

def load_rationale(content_id, rationales_dir):
    caption_filepath = os.path.join(rationales_dir, f"{content_id}.json")
    if not os.path.exists(caption_filepath):
        return None

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
label_mapping = {
    'not_hate': 0,
    'explicit_hate': 1,
    'implicit_hate': 2
}

target_mapping = {
    'None': 0,
    'sex': 1, 
    'race': 2, 
    'religion': 3,
    'nationality': 4, 
    'disability': 5, 
    'org': 6, 
    'political': 6, 
    'others': 6, 
}
 
objs = []
num_negatives = 0
for annot in annotations:
    rationale = load_rationale(annot['ID'], RATIONALE_DIR)
    if rationale:
        obj = {}
        obj["post"] = annot['post']
        obj["class"] = label_mapping[annot['class']]
        obj["class_binarized"] = 1 if label_mapping[annot['class']] >= 1 else 0
        
        if annot['target_categories']:
            targets = []
            for x in annot['target_categories']:
                targets.append(target_mapping[x])
            obj['target_categories'] = annot['target_categories']
            obj['target_categories_mapped'] = targets
        else:
            obj['target_categories'] = None
            if obj['class_binarized'] == 0:
                obj['target_categories_mapped'] = [0]
            else:
                obj['target_categories_mapped'] = [-1]
                num_negatives += 1

        obj["mistral_instruct_statement"] = rationale
        objs.append(obj)

print(num_negatives)
print(json.dumps(obj, indent=2))
with open(OUTPUT_FILEPATH, "w+") as f:
    for obj in objs:
        json.dump(obj, f)
        f.write("\n")