
import tqdm
import os
import json
import argparse
import pandas as pd
import random
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_inference_dataset, load_support_dataset


# from ..matching.tfidf_wrapper import get_top_k_similar as tfidf_sampler
# from ..matching.bm25_wrapper import get_top_k_similar as bm25_sampler
# from ..matching.clip_wrapper import get_top_k_similar as clip_sampler
# from ..matching.sift_wrapper import get_top_k_similar as sift_sampler

from tfidf_wrapper import get_top_k_similar as tfidf_sampler
from bm25_wrapper import get_top_k_similar as bm25_sampler
from clip_wrapper import get_top_k_similar as clip_sampler
# from sift_wrapper import get_top_k_similar as sift_sampler


INTRODUCTION = """## Definition of Hate Speech:
Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

## Instruction
Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful."

## Demonstration Examples

"""

EXAMPLE_TEMPLATE = """### Example {index}
Content:{content}
Answer: {answer}

"""

EXAMPLE_TEMPLATE_WITH_RATIONALE = """### Example {index}
Content:{content}
Answer: {answer}
Rationale: {rationale}

"""

QUESTION_TEMPLATE = """## Task: Evaluate the following content and respond with either "Hateful" or "Not Hateful" based on the provided definition of hate speech.

Content: {content}
Answer: 
"""

def prepare_inputs(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    messages = []

    if use_demonstrations:

        if demonstration_selection == "random":
            samples = random.sample(support_annots, k)

        if demonstration_selection == "tf-idf":
            similar_entries = tfidf_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [support_annots[index] for index in similar_entry_indices]
            
        if demonstration_selection == "bm-25":
            similar_entries = bm25_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [support_annots[index] for index in similar_entry_indices]

        if demonstration_selection == "clip":
            corpus_annotations = [annotation for annotation in support_annots if 'multimodal_record' in annotation]
            similar_entries = clip_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [corpus_annotations[index] for index in similar_entry_indices]

        if demonstration_selection == "sift":
            corpus_annotations = [annotation for annotation in support_annots if 'multimodal_record' in annotation]
            similar_entries = sift_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [corpus_annotations[index] for index in similar_entry_indices]

        formatted_examples = []
        formatted_examples.append(INTRODUCTION)
        if demonstration_distribution == "equal":
            pass
        else:
            for index, s in enumerate(samples):
                answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
                if "rationale" in s.keys():
                    example = EXAMPLE_TEMPLATE_WITH_RATIONALE.format(index=index+1, content=s['content'], rationale=s['rationale'], answer=answer)
                else:
                    example = EXAMPLE_TEMPLATE.format(index=index+1, content=s['content'], answer=answer)
                formatted_examples.append(example)

    question = QUESTION_TEMPLATE.format(content=content['content'])
    formatted_examples.append(question)
    

    joined_examples = "".join(formatted_examples)
    prompt = [{"role": "user", "content": joined_examples}]
    return prompt

def main(
    model_id,
    annotation_filepath,
    caption_dir,
    features_dir,
    result_dir,
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
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir,features_dir)
    support_annots = []
    for filepath, support_caption_dir, support_feature_dir in zip(support_filepaths, support_caption_dirs, support_feature_dirs):
        annots = load_support_dataset(filepath, support_caption_dir, support_feature_dir)
        support_annots += annots
    
    if "rationale" not in sim_matrix_filepath:
        for annot in support_annots:
            annot.pop("rationale", None)

    with open(sim_matrix_filepath, 'rb') as f:
        sim_matrix = np.load(f)
        labels = np.load(f)
        # target_classes=np.load(f, allow_pickle=True)

    os.makedirs(result_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
            messages = prepare_inputs(
                annot,
                idx,
                use_demonstration,
                demonstration_selection,
                demonstration_distribution,
                support_annots,
                sim_matrix,
                labels,
                shots
            )
            

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
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

