import os
import json
import ast
import pandas as pd

# Reconstruct the important attributes
STAGE1_FILEPATH = "/mnt/data1/datasets/hatespeech/sbicv2/originals/SBIC.v2.agg.trn.csv"

df = pd.read_csv(STAGE1_FILEPATH)
# print(df['targetCategory'].value_counts())
print(df.shape)
print(df.head())
print(df.columns)

target_categories = set(df['targetCategory'].unique())
target_categories = [ast.literal_eval(x) for x in list(target_categories)]

unique_categories = []
for x in target_categories:
    unique_categories += x
unique_categories = set(unique_categories)
print(unique_categories)

categories_mapping = {
    'gender': 'sex', 
    'race': "racial", 
    'victim': "others", 
    'disabled': "disability", 
    'social': "others", 
    'culture': "others", 
    'body': "others"
}

