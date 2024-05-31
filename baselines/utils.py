import os
import json

def load_caption(img_filename, caption_dir):
    filename, _ = os.path.splitext(img_filename)
    caption_filepath = os.path.join(caption_dir, f"{filename}.json")
    with open(caption_filepath) as f:
        d = json.load(f)

    return d['caption']

def load_inference_dataset(annotation_filepath, caption_dir, content_template):
    annotations = []
    with open(annotation_filepath) as f:
        for line in f:
            annotations.append(json.loads(line))

    processed_annotations = []
    for annot in annotations:
        obj = {}
        
        if "fhm" in annotation_filepath:
            obj["img"] = os.path.basename(annot['img'])
            obj["text"] = os.path.basename(annot['text'])
            obj["label"] = 1 if annot['gold_hate'][0] == 'hateful' else 0

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = content_template.format(caption=obj['caption'], text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['caption']} {obj['text']}"

        if "mami" in annotation_filepath:
            obj["img"] = annot['file_name']
            obj["text"] = annot['Text Transcription']
            obj["label"] = annot['misogynous']

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = content_template.format(caption=obj['caption'], text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['caption']} {obj['text']}"
        
        processed_annotations.append(obj)

    return processed_annotations

def load_support_dataset(annotation_filepath, caption_dir, content_template):
    annotations = []
    with open(annotation_filepath) as f:
        for line in f:
            annotations.append(json.loads(line))

    processed_annotations = []
    for annot in annotations:
        obj = {}
        
        if "latent_hatred" in annotation_filepath:
            obj["img"] = "N/A"
            obj["text"] = annot['post']
            obj["label"] = annot['class']

            obj["caption"] = "N/A"
            obj["content"] = content_template.format(text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['text']}"

            obj["rationale"] = annot['mistral_instruct_statement']

        if "mmhs" in annotation_filepath.lower():
            obj["img"] = f"{annot['id']}.jpg"
            obj["text"] = annot['tweet_text']
            obj["label"] = annot['label']

            obj["caption"] = load_caption(obj['img'], caption_dir)
            obj["content"] = content_template.format(caption=obj['caption'], text=obj['text'])
            obj["content_for_retrieval"] = f"{obj['caption']} {obj['text']}"

            obj["rationale"] = annot['mistral_instruct_statement']

        processed_annotations.append(obj)

    return processed_annotations