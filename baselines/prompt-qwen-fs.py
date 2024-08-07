
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

from matching.tfidf_wrapper import get_top_k_similar as tfidf_sampler
from matching.bm25_wrapper import get_top_k_similar as bm25_sampler

from prompt_utils import (
    SYSTEM_PROMPT,
    INTRODUCTION,
    INTRODUCTION_WITHOUT_INSTRUCTIONS,
    EXAMPLE_TEMPLATE_WITH_RATIONALE,
    QUESTION_TEMPLATE,
    ANSWER_MULTI_TURN_TEMPLATE,
    QUESTION_MULTI_TURN_TEMPLATE
)


def prepare_inputs(content, content_idx, prompt_format, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    if prompt_format == "system_prompt_demonstrations":
        prepare_fn = prepare_inputs_system_demonstrations
    elif prompt_format == "system_prompt":
        prepare_fn = prepare_inputs_system
    elif prompt_format == "single_prompt":
        prepare_fn = prepare_inputs
    elif prompt_format == "multi_turn_prompt":
        prepare_fn = prepare_inputs_system_conversational
    else:
        raise NotImplementedError(f"prompt format '{prompt_format}' not implemented.")
    
    return prepare_fn(
        content,
        content_idx, 
        use_demonstrations,
        demonstration_selection,
        demonstration_distribution,
        support_annots,
        sim_matrix,
        labels,
        k
    )

def prepare_inputs_single_prompt(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    messages = []

    if use_demonstrations:

        if demonstration_selection == "random":
            similar_entry_indices = sim_matrix[content_idx][:k]
            samples = [support_annots[index] for index in similar_entry_indices]

        if demonstration_selection == "tf-idf":
            similar_entries = tfidf_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [support_annots[index] for index in similar_entry_indices]
            
        if demonstration_selection == "bm-25":
            similar_entries = bm25_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [support_annots[index] for index in similar_entry_indices]

        formatted_examples = []
        formatted_examples.append(INTRODUCTION)
        if demonstration_distribution == "equal":
            pass
        else:
            for index, s in enumerate(samples):
                answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
                example = EXAMPLE_TEMPLATE_WITH_RATIONALE.format(index=index+1, content=s['content'], rationale=s['rationale'], answer=answer)
                formatted_examples.append(example)

    question = QUESTION_TEMPLATE.format(content=content['content'])
    formatted_examples.append(question)
    

    joined_examples = "".join(formatted_examples)
    prompt = [{"role": "user", "content": joined_examples}]

    
    for obj in prompt:
        print("### " + obj['role'].upper())    
        print(obj['content'])
    exit()
    return prompt

def prepare_inputs_system(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    if use_demonstrations:
        if demonstration_selection == "tf-idf":
            similar_entries = tfidf_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [support_annots[index] for index in similar_entry_indices]
        else:
            raise NotImplementedError(f"demonstration selection '{demonstration_selection}' not found")

        msg = [INTRODUCTION_WITHOUT_INSTRUCTIONS]
        if demonstration_distribution == "equal":
            pass
        else:
            for index, s in enumerate(samples):
                answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
                example = EXAMPLE_TEMPLATE_WITH_RATIONALE.format(
                    index=index+1, 
                    content=s['content'], 
                    rationale=s['rationale'], 
                    answer=answer
                )
                msg.append(example)

    question = QUESTION_TEMPLATE.format(content=content['content'])
    msg.append(question)
    

    joined_examples = "\n".join(msg)
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": joined_examples}
    ]
    
    return prompt

def prepare_inputs_system_conversational(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    prompt_conv = []
    if use_demonstrations:
        if demonstration_selection == "tf-idf":
            similar_entries = tfidf_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [support_annots[index] for index in similar_entry_indices]
        else:
            raise NotImplementedError(f"demonstration selection '{demonstration_selection}' not found")

        if demonstration_distribution == "equal":
            pass
        else:
            for index, s in enumerate(samples):
                answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
                qn = QUESTION_MULTI_TURN_TEMPLATE.format(content=s['content'])
                ans = ANSWER_MULTI_TURN_TEMPLATE.format(rationale=s['rationale'], answer=answer)
                prompt_conv.append({"role": "user", "content": qn})
                prompt_conv.append({"role": "assistant", "content": ans})

    qn = QUESTION_MULTI_TURN_TEMPLATE.format(content=content['content'])
    prompt_conv.append({"role": "user", "content": qn})
    
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + prompt_conv 

    
    return prompt

def prepare_inputs_system_demonstrations(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    system_prompt = [SYSTEM_PROMPT, INTRODUCTION_WITHOUT_INSTRUCTIONS]
    prompt_conv = []
    if use_demonstrations:
        if demonstration_selection == "tf-idf":
            similar_entries = tfidf_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
            similar_entry_indices = [entry[0] for entry in similar_entries]
            samples = [support_annots[index] for index in similar_entry_indices]
        else:
            raise NotImplementedError(f"demonstration selection '{demonstration_selection}' not found")

        for index, s in enumerate(samples):
            answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
            example = EXAMPLE_TEMPLATE_WITH_RATIONALE.format(
                index=index+1, 
                content=s['content'], 
                rationale=s['rationale'], 
                answer=answer
            )
            system_prompt.append(example)

    qn = QUESTION_MULTI_TURN_TEMPLATE.format(content=content['content'])
    
    system_prompt = "\n".join(system_prompt)
    prompt_conv.append({"role": "user", "content": qn})
    
    prompt = [
        {"role": "system", "content": system_prompt.strip()}
    ] + prompt_conv 

    
    # for obj in prompt:
    #     print("### " + obj['role'].upper())    
    #     print(obj['content'])
    # exit()
    return prompt

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
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir,features_dir)
    support_annots = []
    for filepath, support_caption_dir, support_feature_dir in zip(support_filepaths, support_caption_dirs, support_feature_dirs):
        annots = load_support_dataset(filepath, support_caption_dir, support_feature_dir)
        support_annots += annots

    with open(sim_matrix_filepath, 'rb') as f:
        sim_matrix = np.load(f)
        labels = np.load(f)
        # target_classes=np.load(f, allow_pickle=True)

    os.makedirs(result_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
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
                prompt_format,
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
                max_new_tokens=10,
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
        response_text = response_text.replace("answer:", "").strip()

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
    parser.add_argument("--prompt_format", choices=["system_prompt", "system_prompt_demonstrations", "single_prompt", "multi_turn_prompt"])
    parser.add_argument("--demonstration_selection", choices=["random", "tf-idf", "bm-25"])
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

