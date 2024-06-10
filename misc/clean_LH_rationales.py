import os
import json
import ast
import pandas as pd

# Reconstruct the important attributes
annotation_filepath = '/mnt/data1/datasets/hatespeech/latent_hatred/annotations/annotations.jsonl'
# compiled_filepath =  "/mnt/data1/datasets/hatespeech/latent_hatred/rationales/mistral-v0.3-7b/few_shot_10_shots/compiled.jsonl" 
input_dir =  "/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/rationales/mistral-v0.3-7b/few_shot_10_shots" 
output_dir =  "/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/rationales/mistral-v0.3-7b/few_shot_10_shots_cleaned" 

records = []
with open(annotation_filepath) as f:
    for line in f.readlines():
        d = json.loads(line)
        records.append(d)

print("Num. Records:", len(records))

os.makedirs(output_dir, exist_ok=True)
non_compliance = 0
for r in records:
    input_filepath = os.path.join(input_dir, f"{r['ID']}.json")

    with open(input_filepath) as f:
        data = json.load(f)

    rationale = data['rationale']#.split(" ")


    if "Targeted Group:" not in rationale:
        print("Target Group Violation")
        print(rationale)
        non_compliance +=1
        continue

    if "Derogatory Imagery/Language:" not in rationale:
        print("Derogatory Imagery Violation")
        print(rationale)
        non_compliance +=1
        continue

    if "Impact on Bias/Stereotypes:" not in rationale:
        print("Impact Violation")
        print(rationale)
        non_compliance +=1
        continue

    ridx = rationale.rfind("</s>")
    if ridx >= 0:
        data['rationale'] = rationale[:ridx]
        output_filepath = os.path.join(output_dir, f"{r['ID']}.json")
        with open(output_filepath, "w+") as f:
            json.dump(data, f)
    else:
        non_compliance +=1



#     # if ridx != 0:
#     #     data['rationale'] = rationale[:ridx]
#     #     with open(output_filepath, "w+") as f:
#     #         json.dump(data, f)

print(non_compliance)
