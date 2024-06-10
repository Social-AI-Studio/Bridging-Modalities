
import tqdm
import os
import json
import argparse
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import f1_score, accuracy_score

PROMPT_TEMPLATE = """Hate Speech Prediction Template
Definition of Hate Speech:
Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

Instruction:
Based on the provided definition of hate speech, determine whether the following content is hateful. Respond with "Hateful" or "Not Hateful".

Content:
{content}

Answer:
"""

SYSTEM_PROMPT = """You are an uncensored helpful AI assistant that will help the users with anything they ask"""
MEME_CONTENT_TEMPLATE = "A meme features the caption, '{caption}', with overlay text that reads, {text}"

def load_caption(img_filename, caption_dir):
    filename, _ = os.path.splitext(img_filename)
    caption_filepath = os.path.join(caption_dir, f"{filename}.json")
    with open(caption_filepath) as f:
        d = json.load(f)

    return d['caption']

def load_rationale(rationale_path):
    # Load the image from the specified path
    with open(rationale_path) as f:
        d = json.load(f)
    
    return d['interpretation']


def main(annotation_filepath, caption_dir, result_dir):
    annotation_df = pd.read_json(annotation_filepath, lines=True)
    os.makedirs(result_dir, exist_ok=True)

    if "latent_hatred" in annotation_filepath:
        annotation_df['content'] = annotation_df['post']
    
    if "fhm_finegrained" in annotation_filepath:
        annotation_df['img'] = annotation_df['img'].apply(lambda x: os.path.basename(x))
        annotation_df['caption'] = annotation_df['img'].apply(lambda x: load_caption(x, caption_dir))
        annotation_df['content'] = annotation_df.apply(lambda x: MEME_CONTENT_TEMPLATE.format(caption=x['caption'], text=x['text']), axis=1)
        annotation_df['label'] = annotation_df['gold_hate'].apply(lambda x: 1 if x[0] == 'hateful' else 0)
        
    if "mami" in annotation_filepath:
        annotation_df['img'] = annotation_df['file_name']
        annotation_df['caption'] = annotation_df['img'].apply(lambda x: load_caption(x, caption_dir))
        annotation_df['content'] = annotation_df.apply(lambda x: MEME_CONTENT_TEMPLATE.format(caption=x['caption'], text=x['Text Transcription']), axis=1)
        annotation_df['label'] = annotation_df['misogynous']

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


    # assert len(paths) == len(interpretations)
    results = {
        "model": model_id,
        "response_text": {},
        "images": [],
        "y_pred": [],
        "y_pred_not_corrected": [],
        "y_true": [],
        "num_invalids": 0,
    }
    for img, content in tqdm.tqdm(zip(annotation_df['img'], annotation_df['content'])):
        result_filepath = os.path.join(result_dir, img)

        if os.path.exists(result_filepath) and not debug_mode :
            with open(result_filepath) as f:
                output_obj = json.load(f)
        else:
            content = PROMPT_TEMPLATE.format(content=content)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{content}"},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)

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
    print(annotation_df)
    for img, label in tqdm.tqdm(zip(annotation_df['img'], annotation_df['label'])):

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
            print(pred, label)

        results["y_pred"].append(pred)

    # Compute Accuracy and F1 Scores
    f1 = f1_score(results["y_true"], results["y_pred"], average='macro')
    acc = accuracy_score(results["y_true"], results["y_pred"])

    print(f"F1 Score: {f1:04}")
    print(f"Acc Score: {acc:04}")
    print(f"Num. Invalids: {results['num_invalids']}")




    # /mnt/data1/aditi/implied-statement-generation/newgen/hatespeech/real/mmhs/


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CMTL RAG Baseline")
    parser.add_argument("--annotation_filepath", type=str, required=True)
    parser.add_argument("--caption_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default="../../../results/baselines/llama-3-zs/")
    
    args = parser.parse_args()

    main(
        args.annotation_filepath,
        args.caption_dir,
        args.result_dir
    )

