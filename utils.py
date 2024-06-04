import os
import json
import pickle

MEME_CONTENT_TEMPLATE = "A meme with the caption, '{caption}', and overlay text that reads, {text}"
POST_CONTENT_TEMPLATE= "A post containing '{text}'"

def load_caption(img_filename, caption_dir):
    filename, _ = os.path.splitext(img_filename)
    caption_filepath = os.path.join(caption_dir, f"{filename}.json")
    with open(caption_filepath) as f:
        d = json.load(f)

    return d['caption']

def load_features(features_dir):
    data_list = []

    # Iterate over all files in the folder
    for filename in os.listdir(features_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(features_dir, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_list.append(data)

    return data_list

def load_inference_dataset(annotation_filepath, caption_dir, features_dir):
    annotations = []
    with open(annotation_filepath) as f:
        for line in f:
            annotations.append(json.loads(line))

    if features_dir:
        features = load_features(features_dir)

    processed_annotations = []
    for index, annot in enumerate(annotations):
        obj = {}
        
        if "fhm" in annotation_filepath:
            obj["img"] = os.path.basename(annot['img'])
            obj["text"] = annot['text']
            obj["label"] = 1 if annot['gold_hate'][0] == 'hateful' else 0

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['caption']} {obj['text']}"

            if features_dir:
                obj["features"] = features[index]

        if "mami" in annotation_filepath:
            obj["img"] = annot['file_name']
            obj["text"] = annot['Text Transcription']
            obj["label"] = annot['misogynous']

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['caption']} {obj['text']}"

            if features_dir:
                obj["features"] = features[index]

        processed_annotations.append(obj)

    return processed_annotations

def load_support_dataset(annotation_filepath, caption_dir, features_dir):
    annotations = []
    with open(annotation_filepath) as f:
        for line in f:
            annotations.append(json.loads(line))

    features = None
    if features_dir is not None and features_dir != "":
        features = load_features(features_dir)

    processed_annotations = []
    for index, annot in enumerate(annotations):
        obj = {}
        
        if "latent_hatred" in annotation_filepath:
            obj["img"] = "N/A"
            obj["text"] = annot['post']
            obj["label"] = annot['class']

            obj["caption"] = "N/A"
            obj["content"] = POST_CONTENT_TEMPLATE.format(text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['text']}"

            obj["rationale"] = annot['mistral_instruct_statement']
            
        if "mmhs" in annotation_filepath.lower():
            obj["img"] = f"{annot['id']}.jpg"
            obj["text"] = annot['tweet_text']
            obj["label"] = 0 if annot['label'] == "not_hateful" else 1

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['caption']} {obj['text']}"

            obj["rationale"] = annot['mistral_instruct_statement']

            if features_dir is not None and features_dir != "":
                obj["features"] = features[index]
            
        if "misogynistic_meme" in annotation_filepath.lower():
            obj["img"] = annot['img']
            obj["text"] = annot['text']
            obj["label"] = annot['label']

            obj["caption"] = annot['caption']
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['caption']} {obj['text']}"

            obj["rationale"] = annot['rationale']
            
            if features_dir is not None and features_dir != "":
                obj["features"] = features[index]
            
        processed_annotations.append(obj)

    return processed_annotations