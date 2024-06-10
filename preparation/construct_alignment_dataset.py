import os
import json

FHM_FILEPATH = "/mnt/data1/datasets/memes/fhm_finegrained/annotations/train.json"
OUTPUT_FILEPATH = "/mnt/data1/datasets/memes/fhm_finegrained/projects/CMTL-RAG/annotations/alignment.jsonl"

records = []
with open(FHM_FILEPATH) as f:
    for line in f:
        d = json.loads(line)
        records.append(d)

# Group records by PC
records = [x for x in records if len(x['gold_pc']) == 1]
unique_pcs = set([x['gold_pc'][0] for x in records])

pc_records = {
    'nationality': [], 
    'pc_empty': [], 
    'disability': [], 
    'religion': [], 
    'race': [], 
    'sex': []
}
for r in records:
    pc_records[r['gold_pc'][0]].append(r)

print("Total Records")
for k, v in pc_records.items():
    print(f"Num. Records for {k}:", len(v))
print()

# Sample 32 records for alignment purposes
sampled_records = {
    'nationality': [], 
    'pc_empty': [], 
    'disability': [], 
    'religion': [], 
    'race': [], 
    'sex': []
}

import random
random.seed(42)
for k in sampled_records.keys():
    sampled_records[k] = random.sample(pc_records[k], 32)

print("Sampled Records")
records = []
for k, v in sampled_records.items():
    print(f"Num. Records for {k}:", len(v))
    records += v

with open(OUTPUT_FILEPATH, "w+") as f:
    for r in records:
        json.dump(r, f)
        f.write("\n")