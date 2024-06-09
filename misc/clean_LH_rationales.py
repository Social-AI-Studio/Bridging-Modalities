import os
import json
import ast
import pandas as pd

# Reconstruct the important attributes
annotation_filepath = '/mnt/data1/datasets/hatespeech/latent_hatred/annotations/annotations.jsonl'
output_dir =  "/mnt/data1/datasets/hatespeech/latent_hatred/rationales/mistral-v0.3-7b/few_shot_10_shots" 

records = []
with open(annotation_filepath) as f:
    for line in f.readlines():
        d = json.loads(line)
        records.append(d)

print("Num. Records:", len(records))

non_compliance = 0
for r in records:
    output_filepath = os.path.join(output_dir, f"{r['ID']}.json")
    with open(output_filepath) as f:
        data = json.load(f)

    rationale = data['rationale']

    if 'Targeted Group' not in rationale:
        non_compliance += 1
        continue

    if 'Derogatory Imagery/Language' not in rationale:
        non_compliance += 1
        continue

    if 'Impact on Bias/Stereotype' not in rationale:
        non_compliance += 1
        continue

print(non_compliance)