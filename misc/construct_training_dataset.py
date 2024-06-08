import os
import json
import pandas as pd

# Reconstruct the important attributes
filename = "train-explanations"
lh_output_filepath = f"/mnt/data1/datasets/hatespeech/latent_hatred/annotations/explanations/truncated/{filename}.jsonl"
lh_rationale_dir = "/mnt/data1/datasets/hatespeech/latent_hatred/rationales/mistral-v0.2-7b/"

lh_stage1_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg1.tsv"
lh_stage1_post_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg1_posts.tsv"
lh_stage2_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg2.tsv"
lh_stage3_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg3.tsv"
lh_dir = f"/mnt/data1/datasets/hatespeech/latent_hatred/annotations"

stage1_df = pd.read_csv(lh_stage1_filepath, delimiter='\t')
stage1_post_df = pd.read_csv(lh_stage1_post_filepath, delimiter='\t')
stage2_df = pd.read_csv(lh_stage2_filepath, delimiter='\t')
stage3_df = pd.read_csv(lh_stage3_filepath, delimiter='\t')

stage1_df.drop('class', axis=1, inplace=True)
stage1_df = pd.concat([stage1_df, stage1_post_df], axis=1)

print("Stage1:", stage1_df.shape)
print("Stage2 (Duplicates):", stage1_df[stage1_df['ID'].duplicated()].shape)
print("Stage1 (Distribution):", stage1_df['class'].value_counts())
print()

print("Stage2:", stage2_df.shape, stage2_df.columns)
print("Stage2 (Duplicates):", stage2_df[stage2_df['ID'].duplicated()].shape)
print()

lh_target_mapping_filepath = '/mnt/data1/mshee/CMTL-RAG/cmdr/lh_target_mapping_new.json'
with open(lh_target_mapping_filepath) as f:
    target_mapping = json.load(f)

# target_values = 
unique_values = set()
for lst in list(target_mapping.values()):
    assert len(lst) >= 1
    for i in lst:
        unique_values.add(i)

print("Number of Targets (Source):", len(target_mapping))
print("Number of Targets (Target):", unique_values)
print()

def get_target_categories(x):
    if x == "None" or isinstance(x, float):
        return ["None"]

    x = x.lower()
    new_lst = target_mapping[x]

    return new_lst

print("Stage3 Dataframe:", stage3_df.shape, stage3_df.columns)
print("Stage3 Dataframe (Duplicates):", stage3_df[stage3_df['ID'].duplicated()].shape)

stage3_df['target_categories'] = stage3_df['target'].apply(lambda x: get_target_categories(x))
stage3_df['target_categories_count'] = stage3_df['target_categories'].apply(len)
print("Stage3 Dataframe (After Processing):", stage3_df['target_categories_count'].value_counts())
print()

# Construct original dataframes
print("Stage 1:", stage1_df.shape, stage1_df.shape)
orig_df = pd.merge(stage1_df, stage2_df, left_on="ID", right_on="ID", how="left")
print("Stage 2:", orig_df.shape, orig_df.columns)

# Map Stage 3 Annotations
orig_df = pd.merge(orig_df, stage3_df, left_on="ID", right_on="ID", how="left")
print("Stage 3:", orig_df.shape, orig_df.columns)

output_filepath = os.path.join(lh_dir, "annotations.jsonl")
orig_df.to_json(output_filepath, orient="records", lines=True)