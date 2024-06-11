import os
import json
import pickle

MEME_CONTENT_TEMPLATE = "A meme with the caption, '{caption}', and overlay text that reads, {text}"
POST_CONTENT_TEMPLATE= "A post containing '{text}'"

fhm_target_mapping = {
    'pc_empty': 0, 
    'sex': 1, 
    'race': 2, 
    'religion': 3,
    'nationality': 4, 
    'disability': 5, 
}

def load_caption(img_filename, caption_dir):
    filename, _ = os.path.splitext(img_filename)
    caption_filepath = os.path.join(caption_dir, f"{filename}.json")
    with open(caption_filepath) as f:
        d = json.load(f)

    return d['caption']

def load_features(features_dir):
    data_dict = {}

    # Iterate over all files in the folder
    for filename in os.listdir(features_dir):
        if filename.endswith('.pkl'):
            # Remove the .pkl extension to get the id
            file_id = filename[:-4]
            file_path = os.path.join(features_dir, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_dict[file_id] = data

    return data_dict


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
            obj["id"] = str(annot["id"])
            if len(obj["id"]) < 5:
                obj["id"] = "0" + obj["id"]
            obj["img"] = os.path.basename(annot['img'])
            obj["text"] = annot['text']
            obj["label"] = 1 if annot['gold_hate'][0] == 'hateful' else 0

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["content_text"] = f"{obj['text']}"
            obj["content_text_caption"] = f"{obj['text']} {obj['caption']}"

            if features_dir is not None and features_dir != "":
                obj["features"] = features[obj["id"]]

            obj["multimodal_record"] = True

            obj["target_categories_mapped"] = [fhm_target_mapping[x] for x in annot["gold_pc"]]

        if "mami" in annotation_filepath:
            obj["id"] = annot["file_name"][:-4]
            obj["img"] = annot['file_name']
            obj["text"] = annot['Text Transcription']
            obj["label"] = annot['misogynous']

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["content_text"] = f"{obj['text']}"
            obj["content_text_caption"] = f"{obj['text']} {obj['caption']}"

            if features_dir is not None and features_dir != "":
                obj["features"] = features[obj["id"]]

            obj["multimodal_record"] = True

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
            obj["label"] = annot['class_binarized']

            obj["caption"] = "N/A"
            obj["content"] = POST_CONTENT_TEMPLATE.format(text=obj['text'])
            obj["content_text"] = f"{obj['text']}"
            obj["context_text_caption"] = f"{obj['text']}"

            obj["rationale"] = annot["mistral_instruct_statement"]
            obj["target_categories_mapped"] = annot["target_categories_mapped"]
            
        if "mmhs" in annotation_filepath.lower():
            obj["id"] = annot["id"]
            obj["img"] = f"{annot['id']}.jpg"
            obj["text"] = annot['tweet_text']
            obj["label"] = 0 if annot['label'] == "not_hateful" else 1

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["context_text"] = f"{obj['text']}"
            obj["context_text_caption"] = f"{obj['text']} {obj['caption']}"

            obj["rationale"] = annot['mistral_instruct_statement']

            if features_dir is not None and features_dir != "":
                obj["features"] = features[obj["id"]]

            obj["multimodal_record"] = True
            
        if "alignment" in annotation_filepath.lower():
            obj["id"] = f"{annot['id']:05}"
            obj["img"] = os.path.basename(annot['img'])
            obj["text"] = annot['text']
            obj["label"] = 1 if annot['gold_hate'][0] == 'hateful' else 0

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["content_text"] = f"{obj['text']}"
            obj["content_text_caption"] = f"{obj['text']} {obj['caption']}"

            if features_dir is not None and features_dir != "":
                obj["features"] = features[obj["id"]]

            obj["rationale"] = annot['mistral_instruct_statement']
            obj["multimodal_record"] = True
            
        if "misogynistic_meme" in annotation_filepath.lower():
            obj["id"] = annot["id"]
            obj["img"] = annot['img']
            obj["text"] = annot['text']
            obj["label"] = annot['label']

            obj["caption"] = annot['caption']
            obj["content"] = MEME_CONTENT_TEMPLATE.format(caption=obj['caption'], text=obj['text'])
            obj["context_text"] = f"{obj['text']}"
            obj["context_text_caption"] = f"{obj['text']} {obj['caption']}"

            obj["rationale"] = annot["mistral_instruct_statement"]
            
            if features_dir is not None and features_dir != "":
                obj["features"] = features[obj["id"]]

            obj["multimodal_record"] = True
            
        processed_annotations.append(obj)

    print(f"{len(processed_annotations)} records have been loaded")
    return processed_annotations