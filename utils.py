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

def load_rationales(folder_path):
    json_dict = {}
    
    # List all files in the directory
    for file_name in os.listdir(folder_path):
        
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            
            # Extract the filename without the ".json" extension
            json_id = os.path.splitext(file_name)[0]
            
            # Open and read each JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                explanation = data.get('rationale')
                
                # Add to dictionary if the explanation exists
                if explanation is not None:
                    json_dict[json_id] = explanation

    return json_dict

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

def load_support_dataset(annotation_filepath, caption_dir, features_dir, rationales_dir):
    annotations = []
    with open(annotation_filepath) as f:
        for line in f:
            annotations.append(json.loads(line))

    features = None
    if features_dir is not None and features_dir != "":
        features = load_features(features_dir)

    rationales = None
    if rationales_dir:
        rationales = load_rationales(rationales_dir)

    processed_annotations = []
    for index, annot in enumerate(annotations):
        obj = {}
        
        if "latent_hatred" in annotation_filepath:
            obj["id"] = annot["ID"]
            obj["img"] = "N/A"
            obj["text"] = annot['post']
            obj["label"] = annot['class_binarized']

            obj["caption"] = "N/A"
            obj["content"] = POST_CONTENT_TEMPLATE.format(text=obj['text'])
            obj["content_text"] = f"{obj['text']}"
            obj["context_text_caption"] = f"{obj['text']}"

            obj["rationale"] = rationales[obj["id"]]
            
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

            obj["rationale"] = annot['rationale']
            
            if features_dir is not None and features_dir != "":
                obj["features"] = features[obj["id"]]

            obj["multimodal_record"] = True
            
        processed_annotations.append(obj)

    print(f"{len(processed_annotations)} records have been loaded")
    return processed_annotations