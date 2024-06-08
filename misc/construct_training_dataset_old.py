import os
import json
import pandas as pd

# Reconstruct the important attributes
filename = "train-explanations"
lh_output_filepath = f"/mnt/data1/datasets/hatespeech/latent_hatred/annotations/explanations/truncated/{filename}.jsonl"
lh_rationale_dir = "/mnt/data1/datasets/hatespeech/latent_hatred/rationales/mistral-v0.2-7b/"

lh_stage1_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/implicit_hate_v1_stg1.tsv"
lh_stage1_post_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/implicit_hate_v1_stg1_posts.tsv"
lh_stage2_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/implicit_hate_v1_stg2.tsv"
lh_stage3_filepath = "/mnt/data1/datasets/hatespeech/latent_hatred/implicit_hate_v1_stg3.tsv"
lh_dir = f"/mnt/data1/datasets/hatespeech/latent_hatred/annotations/explanations"

stage1_df = pd.read_csv(lh_stage1_filepath, delimiter='\t')
stage1_post_df = pd.read_csv(lh_stage1_post_filepath, delimiter='\t')
stage2_df = pd.read_csv(lh_stage2_filepath, delimiter='\t')
stage3_df = pd.read_csv(lh_stage3_filepath, delimiter='\t')


stage1_df.drop('class', axis=1, inplace=True)
stage1_df = pd.concat([stage1_df, stage1_post_df], axis=1)

print("Stage1:", stage1_df.shape)
print("Stage1 Distribution:", stage1_df['class'].value_counts())
print()

print("Stage2:", stage2_df.shape, stage2_df.columns)
print("Stage2 Duplicates:", stage2_df[stage2_df['ID'].duplicated()].shape)
print()

def merge_targets(lst):
    new_lst = []
    special_removal = ["folks", "people", ".", "citizens", "devils"]
    special_mapping = {
        "arabians": "arabs",
        "pakistani people": "pakistanis",
        "jewish": "jews",
        "illegals": "illegal immigrants",
        "blacks": "black",
        "whites": "white",
        "pedophiles": "pedophile",
        "lesbians": "lesbian",
        "asians": "asian",
        "immmigrants": "immigrants",
        "imigrantes": "immigrants",
        'islam followers': 'islamic',
        "anti gay": "gay",
        "gays": "gay",
        "gay": "lgbt",
        "fat women": "women",

    }

    custom_replacement = (
        [['illegal immigrants', 'immigrants'], ['immigrants', 'immigrants']],
        [['immigrants', 'illegal immigrants'], ['immigrants', 'immigrants']],
        [['muslim and jews', 'jews and muslims'], ['muslim and jews', 'muslim and jews']],
        [['muslims', 'islamic'], ['islamic', 'islamic']],
        [['islamic', 'muslims'], ['islamic', 'islamic']],
        [['muslims', 'islamic followers'], ['muslims', 'muslims']],
        [['black', 'black in south africa'], ['black', 'black']],
        [['illegal aliens', 'illegal immigrants'], ['illegal immigrants', 'illegal immigrants']],
        [['bible publishers', 'bible publisher'], ['bible publisher', 'bible publisher']],
        [['black', 'black teens'], ['black', 'black']],
        [['muslims', 'muslims', 'muslims', 'islamic'], ['muslims', 'muslims', 'muslims', 'muslims']],
        [['of color', 'non-white'], ['of color', 'of color']],
        [['islamic', 'islam'], ['islamic', 'islamic']],
        [['illegal immigrant', 'illegal immigrants'], ['illegal immigrants', 'illegal immigrants']],
        [['black', 'kanye west'], ['black', 'black']],
        [['non-white', 'of color'], ['of color', 'of color']],
        [['islamic', 'who are islamic'], ['islamic', 'islamic']],
        [['muslims', 'african muslims'], ['muslims', 'muslims']],
        [['illegal immigrants', 'non-white immigrants', 'non-white immigrants', 'immigrants'], ['immigrants', 'immigrants', 'immigrants', 'immigrants']],
        [['minorities', 'non-white'], ['minorities', 'minorities']],
        [['solmalis', 'somalis'], ['somalis', 'somalis']],
        [['immigrants and their children', 'immigrants'], ['immigrants', 'immigrants']],
        [['gay', 'lgbt'], ['lgbt', 'lgbt']],
        [['gay', 'gays'], ['gays', 'gays']],
        [['white men', 'white males'], ['white men', 'white men']],
        [['immigrants', 'mexicans'], ['immigrants', 'mexicans']],
        [['immigrants', 'californians'], ['immigrants', 'immigrants']],
        [['immigrants', 'latino'], ['immigrants', 'immigrants']],
        [['womens', 'women'], ['women', 'women']],
        [['immigrants', 'illegal'], ['immigrants', 'immigrants']],
        [['immigrants', 'foreign'], ['immigrants', 'immigrants']],
        [['latino/latina immigrants', 'immigrants'], ['immigrants', 'immigrants']],
        [['lgbtq', 'gays'], ['lgbtq', 'lgbtq']],
        [['haitians', 'haitian immigrants'], ['immigrants', 'immigrants']],
        [['of color, black', 'black'], ['black', 'black']],
        [['immigrants', 'third world citizens'], [['immigrants', 'immigrants']]],
        [['immigrants', 'refugees'], ['immigrants', 'immigrants']],
        [['black', 'black conservatives'], ['black', 'black']],
        [['indian', 'indians immigrants'], ['indian', 'indian']],
        [['leftists', 'left'], ['leftists', 'leftists']],
        [['illegal immigrants', 'immigrants, in this case jews'], ['immigrants', 'immigrants']],
        [['irish and polish immigrants', 'colored immigrants'], ['immigrants', 'immigrants']],
        [['black', 'non-white'], ['of color', 'of color']],
        [['black', 'of color'], ['of color', 'of color']],
        [['africans', 'black africans'], ['africans', 'africans']],
        [['indians', 'american indians'], ['indians', 'indians']],
        [['mexicans', 'latino'], ['latino', 'latino']],
        [['white jews', 'jews'], ['jews', 'jews']],
        [['immigrants', 'non-white immigrants'], ['immigrants', 'immigrants']],
        [['mexicans', 'immigrants'], ['immigrants', 'immigrants']],
        [['immigrants', "mexicans"], ['immigrants', 'immigrants']],
        [['gays', 'lgbt'], ['lgbt', 'lgbt']],
        [['immigrants', 'aliens'], ['immigrants', 'immigrants']],
        [['islam', 'islamic'], ['islamic', 'islamic']]
    )
    for x in lst:
        if x == "None" or isinstance(x, float):
            new_lst.append("None")
            continue

        x = x.lower()
        for word in special_removal:
            x = x.replace(word, "").strip()

        for k, v in special_mapping.items():
            x = x.replace(k, v)

        new_lst.append(x)

    for s, t in custom_replacement:
        if new_lst == s:
            new_lst = t

    if len(set(new_lst)) == 1:
        return [new_lst[0]]

    from collections import Counter
    counter = Counter(new_lst)
    counts = counter.most_common(2)
    if counts[0][1] > counts[1][1]:
        return [counts[0][0]]
    # if 

    skipped_lst = [
        ['white', 'black'], 
        ['black', 'white'],
        ['immigrants', 'politicians'],
    ]
    if new_lst not in skipped_lst:
        print(new_lst)
    return new_lst


print("Stage3 Dataframe:", stage3_df.shape, stage3_df.columns)
print("Stage3 Dataframe:", stage3_df.groupby("ID").agg(len)['target'].value_counts())
print("Stage3 Dataframe (Duplicates):", stage3_df[stage3_df['ID'].duplicated()].shape)

stage3_df = stage3_df.groupby("ID").agg(list).reset_index()
stage3_df['target'] = stage3_df['target'].apply(lambda x: merge_targets(x))
stage3_df['target_count'] = stage3_df['target'].apply(len)
print("Stage3 Dataframe:", stage3_df['target_count'].value_counts())
print("Stage3 Dataframe (Duplicates):", stage3_df[stage3_df['ID'].duplicated()].shape)
# print("Stage3 Dataframe (Duplicates):", stage3_df[stage3_df['ID'].duplicated(keep=False)].iloc[40:80])
# print("Stage3 Dataframe (Duplicates):", {v.lower(): "undefined" for v in list(stage3_df[stage3_df['ID'].duplicated(keep=False)]['target'].unique()) if v != "None" and not isinstance(v, float)})
print(stage1_df[stage1_df['ID'] == "466220911675179008"].iloc[0,1])
print()
exit()

# Construct original dataframes
print("Stage 1:", stage1_df.shape)
orig_df = pd.merge(stage1_df, stage2_df, left_on="ID", right_on="ID", how="left")
# orig_df.drop_duplicates("post", inplace=True)
print("Stage 2:", orig_df.shape, orig_df.columns)

# Map Stage 3 Annotations
orig_df = pd.merge(orig_df, stage3_df, left_on="ID", right_on="ID", how="left")
# orig_df.drop_duplicates("post", inplace=True)
# print("Stage 3:", orig_df[orig_df.duplicated(keep=False)], orig_df.columns)
print("Stage 3:", orig_df.shape)

stage1_filepath = os.path.join(lh_dir, "stage1.jsonl")
stage1_df.to_json(stage1_filepath, orient="records", lines=True)

# targets = list((stage3_df['target'].unique()))
# targets = set([x.lower() for x in targets if x != None and isinstance(x, str)])

# targets_mapping = {x: "undefined" for x in targets}
# print(len(targets))
# print(len(targets_mapping))

# with open("lh_target_mapping.json") as f:
#     d = json.load(f)

# print(len(d))
# for k, v in targets_mapping.items():
#     if k in d:
#         targets_mapping[k] = d[k]
        
# print(len([x for x in targets_mapping.values() if x == "undefined"]))
# with open("lh_target_mapping_new.json", "w+") as f:
#     json.dump(targets_mapping, f)
# exit()

# Remap Stage 1 Annotations
output_df = pd.merge(input_df, stage1_df, left_on="post", right_on="post", how="left")
output_df.drop_duplicates("post", inplace=True)
print("Post-Stage 1:", output_df.shape)
print("Post-Stage 1:", output_df['class'].value_counts())

# Map Stage 2 Annotations
output_df = pd.merge(output_df, stage2_df, left_on="ID", right_on="ID", how="left")
output_df.drop_duplicates("post", inplace=True)
print("Post-Stage 2:", output_df.shape)

# Map Stage 3 Annotations
output_df = pd.merge(output_df, stage3_df, left_on="ID", right_on="ID", how="left")
output_df.drop_duplicates("post", inplace=True)
print("Post-Stage 2:", output_df.shape)

# Remap hate
class_mapping = {
    "not_hate": 0,
    "implicit_hate": 1,
    "explicit_hate": 2
}
output_df['class'] = output_df['class'].apply(lambda x: class_mapping[x])

for post_id, rationale in zip(output_df['ID'], output_df['mistral_instruct_statement']):
    output_filepath = os.path.join(lh_rationale_dir, f"{post_id}.json")
    with open(output_filepath, "w+") as f:
        json.dump({"rationale": rationale}, f)


# Remap target
# print(len(set(output_df['target'].tolist())))
# print(set(output_df['target'].tolist()))

output_df.to_json(lh_output_filepath, orient="records", lines=True)