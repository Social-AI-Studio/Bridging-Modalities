import os
import json
import pandas as pd

# Reconstruct the important attributes
STAGE1_FILEPATH = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg1.tsv"
STAGE2_FILEPATH = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg2.tsv"
STAGE3_FILEPATH = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg3.tsv"

STAGE1_POST_FILEPATH = "/mnt/data1/datasets/hatespeech/latent_hatred/originals/implicit_hate_v1_stg1_posts.tsv"

TARGET_MAPPING_FILEPATH = '/mnt/data1/mshee/CMTL-RAG/cmdr/lh_target_mapping_new.json'
OUTPUT_DIR = f"/mnt/data1/datasets/hatespeech/latent_hatred/annotations"

TARGET_VALUES = {
    'sex': 1,
    'race': 2,
    'religion': 3,
    'nationality': 4,
    'disability': 5,
    'None': 6,
    'others': 7
}


def load_stage1_dataset(annot_filepath, post_filepath):
    stage1_df = pd.read_csv(annot_filepath, delimiter='\t')
    stage1_post_df = pd.read_csv(post_filepath, delimiter='\t')
    
    stage1_df.drop('class', axis=1, inplace=True)
    stage1_df = pd.concat([stage1_df, stage1_post_df], axis=1)

    print("Stage1:", stage1_df.shape)
    print("Stage2 (Duplicates):", stage1_df[stage1_df['ID'].duplicated()].shape)
    print("Stage1 (Distribution):", stage1_df['class'].value_counts())
    print()

    return stage1_df

def load_stage2_dataset(annot_filepath):
    stage2_df = pd.read_csv(annot_filepath, delimiter='\t')
    print("Stage2:", stage2_df.shape, stage2_df.columns)
    print("Stage2 (Duplicates):", stage2_df[stage2_df['ID'].duplicated()].shape)
    print()

    return stage2_df

def load_stage3_dataset(filepath, target_mapping, target_values):
    stage3_df = pd.read_csv(filepath, delimiter='\t')
    print("Stage3 Dataframe:", stage3_df.shape, stage3_df.columns)
    print("Stage3 Dataframe (Duplicates):", stage3_df[stage3_df['ID'].duplicated()].shape)

    stage3_df['target_categories'] = stage3_df['target'].apply(
        lambda x: get_target_categories(x, target_mapping)
    )
    stage3_df['target_categories_values'] = stage3_df['target_categories'].apply(
        lambda lst: [
            TARGET_VALUES[x] if x in target_values else TARGET_VALUES['None']
            for x in lst
        ]
    )
    stage3_df['target_categories_count'] = stage3_df['target_categories'].apply(len)
    print("Stage3 Dataframe (After Processing):", stage3_df['target_categories_count'].value_counts())
    print("Stage3 Dataframe (After Processing):", stage3_df['target_categories_values'].value_counts())
    print()

    return stage3_df

def load_target_mapping(target_mapping_filepath):
    with open(target_mapping_filepath) as f:
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

    return target_mapping

def get_target_categories(x, target_mapping):
    if x == "None" or isinstance(x, float):
        return ["None"]

    x = x.lower()
    new_lst = target_mapping[x]

    return new_lst

target_mapping = load_target_mapping(TARGET_MAPPING_FILEPATH)

stage1_df = load_stage1_dataset(STAGE1_FILEPATH, STAGE1_POST_FILEPATH)
stage2_df = load_stage2_dataset(STAGE2_FILEPATH)
stage3_df = load_stage3_dataset(STAGE3_FILEPATH, target_mapping, TARGET_VALUES)

# Construct original dataframes
print("Stage 1:", stage1_df.shape, stage1_df.shape)
orig_df = pd.merge(stage1_df, stage2_df, left_on="ID", right_on="ID", how="left")
print("Stage 2:", orig_df.shape, orig_df.columns)
orig_df = pd.merge(orig_df, stage3_df, left_on="ID", right_on="ID", how="left")
print("Stage 3:", orig_df.shape, orig_df.columns)

output_filepath = os.path.join(OUTPUT_DIR, "annotations.jsonl")
orig_df.to_json(output_filepath, orient="records", lines=True)