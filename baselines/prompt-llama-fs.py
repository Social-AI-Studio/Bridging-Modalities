import torch
import tqdm
import os
import json
import argparse
import pandas as pd
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_inference_dataset, load_support_dataset


# from ..matching.tfidf_wrapper import get_top_k_similar as tfidf_sampler
# from ..matching.bm25_wrapper import get_top_k_similar as bm25_sampler
# from ..matching.clip_wrapper import get_top_k_similar as clip_sampler
# from ..matching.sift_wrapper import get_top_k_similar as sift_sampler

from matching.tfidf_wrapper import get_top_k_similar as tfidf_sampler
from matching.bm25_wrapper import get_top_k_similar as bm25_sampler
# from matching.clip_wrapper import get_top_k_similar as clip_sampler
# from sift_wrapper import get_top_k_similar as sift_sampler

from prompt_utils import (
    SYSTEM_PROMPT,
    INTRODUCTION,
    INTRODUCTION_WITHOUT_INSTRUCTIONS,
    EXAMPLE_TEMPLATE_WITH_RATIONALE,
    QUESTION_TEMPLATE,
    ANSWER_MULTI_TURN_TEMPLATE,
    QUESTION_MULTI_TURN_TEMPLATE
)

from fs_utils import prepare_inputs

def main(
    model_id,
    annotation_filepath,
    caption_dir,
    features_dir,
    result_dir,
    prompt_format,
    use_demonstration,
    demonstration_selection,
    demonstration_distribution,
    support_filepaths,
    support_caption_dirs,
    support_feature_dirs,
    sim_matrix_filepath,
    debug_mode,
    shots
    ):
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir,features_dir, None)
    support_annots = []
    for filepath, support_caption_dir, support_feature_dir in zip(support_filepaths, support_caption_dirs, support_feature_dirs):
        annots = load_support_dataset(filepath, support_caption_dir, support_feature_dir, None)
        support_annots += annots
    
    with open(sim_matrix_filepath, 'rb') as f:
        sim_matrix = np.load(f)
        labels = np.load(f)

    os.makedirs(result_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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

    for idx, annot in enumerate(tqdm.tqdm(inference_annots)):
        img, content = annot['img'], annot['content']
        id = annot["id"]
        file_extension = ".json"
        filename = id + file_extension
        result_filepath = os.path.join(result_dir, filename)

        if os.path.exists(result_filepath) and not debug_mode:
            with open(result_filepath) as f:
                output_obj = json.load(f)
        else:
            content = prepare_inputs(
                annot,
                idx, 
                prompt_format,
                use_demonstration,
                demonstration_selection,
                demonstration_distribution,
                support_annots,
                sim_matrix,
                labels,
                shots
            )
            messages = [
                {"role": "user", "content": f"{content}"},
            ]
            

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                num_beams=1
            )

            response = outputs[0][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            response_text= response_text.replace('\n', '')
            response_text= response_text.replace('**', '')
            
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
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, required=True)

    parser.add_argument("--use_demonstrations", action="store_true")
    parser.add_argument("--prompt_format", choices=["system_prompt", "single_prompt", "multi_turn_prompt"])
    parser.add_argument("--demonstration_selection", choices=["random", "tf-idf", "bm-25", "clip", "sift"])
    parser.add_argument("--demonstration_distribution", choices=["equal", "top-k"])
    parser.add_argument("--support_filepaths", nargs='+')
    parser.add_argument("--support_caption_dirs", nargs='+')
    parser.add_argument("--support_feature_dirs", nargs='+')
    parser.add_argument("--sim_matrix_filepath", type=str, required=True)
    parser.add_argument("--shots", type=int, required=True)

    args = parser.parse_args()

    main(
        args.model_id,
        args.annotation_filepath,
        args.caption_dir,
        args.feature_dir,
        args.result_dir,
        args.prompt_format,
        args.use_demonstrations,
        args.demonstration_selection,
        args.demonstration_distribution,
        args.support_filepaths,
        args.support_caption_dirs,
        args.support_feature_dirs,
        args.sim_matrix_filepath,
        args.debug_mode,
        args.shots
    )

