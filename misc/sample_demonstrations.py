import os
import json

lh_dir = f"/mnt/data1/datasets/hatespeech/latent_hatred/annotations"
lh_filepath = os.path.join(lh_dir, "annotations.jsonl")

# Load dataset
data = []
with open(lh_filepath) as f:
    for line in f:
        d = json.loads(line)
        data.append(d)

print(len(data))

values = set()
for d in data:
    values.add(d['class'])

print(values)

records = {idx:[] for idx in range(6)}
for d in data:
    values = d['target_categories_values']
    if values == None or len(values) != 1:
        continue
    
    v = values[0]

    # skip records that are either political or others
    if v >= 6:
        continue 

    records[v].append(d)

import random
random.seed(42)

sampled_records = []
for k,v in records.items():
    if k == 0:
        continue
    
    if k in [2, 5]:
        sampled_records += random.sample(v, 8)
    else:
        sampled_records += random.sample(v, 4)

print(len(sampled_records))
output_filepath = os.path.join(lh_dir, "sampled_demonstrations.jsonl")
with open(output_filepath, "w+") as f:
    for r in sampled_records:
        json.dump(r, f)
        f.write("\n")

