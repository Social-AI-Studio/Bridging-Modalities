import os
import json

lh_dir = f"/mnt/data1/datasets/hatespeech/latent_hatred/annotations"
lh_filepath = os.path.join(lh_dir, "annotations.jsonl")

# Load dataset
data = []
with open(lh_filepath) as f:
    for line in f:
        d = json.loads(line)

        # print(d['class'])
        if d['class'] == "not_hate":
            data.append(d)

print(len(data))

import random
random.seed(42)

sampled_records = random.sample(data, 10)

print(len(sampled_records))
output_filepath = os.path.join(lh_dir, "sampled_demonstrations_nonhate.jsonl")
with open(output_filepath, "w+") as f:
    for r in sampled_records:
        json.dump(r, f)
        f.write("\n")

