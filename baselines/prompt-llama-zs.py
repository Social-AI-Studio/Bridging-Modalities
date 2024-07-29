
import tqdm
import os
import json
import argparse
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, accuracy_score
from utils import load_inference_dataset, load_support_dataset

INFERENCE_PROMPT_TEMPLATE = """Hate Speech Prediction Template
Definition of Hate Speech:
Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

Instruction:
Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful."

Content:
{content}
Answer:
"""

def main(model_id, annotation_filepath, caption_dir, result_dir, debug_mode):
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None, None)
    
    os.makedirs(result_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/mnt/data1/aditi/hf/new_cache_dir/", device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/mnt/data1/aditi/hf/new_cache_dir/", device_map="cuda")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="cuda",
    )

    results = {
        "model": model_id,
        "response_text": {},
        "images": [],
        "y_pred": [],
        "y_pred_not_corrected": [],
        "y_true": [],
        "num_invalids": 0,
    }
    if debug_mode:
        inference_annots = inference_annots[:5]

    for annot in tqdm.tqdm(inference_annots):
        img, content = annot['img'], annot['content']
        id = annot["id"]
        file_extension = ".json"
        filename = id + file_extension
        result_filepath = os.path.join(result_dir, filename)

        if os.path.exists(result_filepath) and not debug_mode :
            with open(result_filepath) as f:
                output_obj = json.load(f)
        else:
            content = INFERENCE_PROMPT_TEMPLATE.format(content=content)
            messages = [
                {"role": "user", "content": f"{content}"},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=10,
                do_sample=False,
                num_beams=1
            )
            response_text=outputs[0]["generated_text"][-1]['content']

            output_obj = {
                "img": img,
                "model": model_id,
                "response_text": response_text,
                "content": content
            }

            with open(result_filepath, "w+") as f:
                json.dump(output_obj, f)


        results["response_text"][output_obj['img']] = output_obj['response_text']

    # Answer Processing
    for annot in tqdm.tqdm(inference_annots):
        img, label = annot['img'], annot['label']
        results["images"].append(img)
        results["y_true"].append(label)

        response_text = results["response_text"][img].lower()

        pred = -1
        if response_text.startswith("not hateful"):
            pred = 0
        
        if response_text.startswith("hateful"):
            pred = 1

        results["y_pred_not_corrected"].append(pred)

        if pred == -1:
            if label == 1:
                pred = 0
            else:
                pred = 1
            results["num_invalids"] += 1

        results["y_pred"].append(pred)

    # Compute Accuracy and F1 Scores
    f1 = f1_score(results["y_true"], results["y_pred"], average='macro')
    acc = accuracy_score(results["y_true"], results["y_pred"])

    print(f"F1 Score: {f1:04}")
    print(f"Acc Score: {acc:04}")
    print(f"Num. Invalids: {results['num_invalids']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CMTL RAG Baseline")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--annotation_filepath", type=str, required=True)
    parser.add_argument("--caption_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, required=True)
    
    args = parser.parse_args()

    main(
        args.model_id,
        args.annotation_filepath,
        args.caption_dir,
        args.result_dir,
        args.debug_mode
    )

