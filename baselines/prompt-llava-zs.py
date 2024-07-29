import re
import tqdm
import os
import torch
import json
import argparse
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from utils import load_inference_dataset, load_support_dataset, load_images
from prompt_utils import ZS_SINGLE_TURN_PROMPT

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

# Reference: https://github.dev/haotian-liu/LLaVA/pull/1502/files

def replace_image_tokens(qs, mm_use_im_start_end):
    # qs = qs_orig.copy()
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    print(IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER in qs)
    if IMAGE_PLACEHOLDER in qs:
        if mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    return qs

def main(model_path, model_base, annotation_filepath, caption_dir, image_dir, result_dir, conv_mode, debug_mode):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    # Data
    print(image_dir)
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None, image_dir)

    # Conv Mode
    if "llama-2" in model_name.lower():
        detected_conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        detected_conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        detected_conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        detected_conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        detected_conv_mode = "mpt"
    else:
        detected_conv_mode = "llava_v0"

    if conv_mode is not None and detected_conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                detected_conv_mode, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = detected_conv_mode

    results = {
        "model": model_path,
        "response_text": {},
        "images": [],
        "y_pred": [],
        "y_pred_not_corrected": [],
        "y_true": [],
        "num_invalids": 0,
    }
    
    for idx, annot in tqdm.tqdm(enumerate(inference_annots)):
        img, img_path, content = annot['img'], annot['img_path'], annot['content_llava']
        filename = f"{annot['id']}.json"
        result_filepath = os.path.join(result_dir, filename)

        if os.path.exists(result_filepath) and not debug_mode:
            with open(result_filepath) as f:
                output_obj = json.load(f)

            output_obj['img'] = os.path.basename(output_obj['img'])

            with open(result_filepath, "w+") as f:
                json.dump(output_obj, f)
        else:
            # Prepare Conv
            qs = ZS_SINGLE_TURN_PROMPT.format(content=content)
            qs = replace_image_tokens(qs, model.config.mm_use_im_start_end)

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Parse Images
            images = load_images([img_path])
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=32,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            output_obj = {
                "img": img,
                "model": model_path,
                "response_text": outputs,
                "content": content
            }

            with open(result_filepath, "w+") as f:
                json.dump(output_obj, f)


        results["response_text"][output_obj['img']] = output_obj['response_text']

    # print("YO")
    # print(results)
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
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--debug_mode", type=bool, default=False)
    parser.add_argument("--annotation_filepath", type=str, required=True)
    parser.add_argument("--caption_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, required=True)
    
    args = parser.parse_args()

    main(
        args.model_path,
        args.model_base,
        args.annotation_filepath,
        args.caption_dir,
        args.image_dir,
        args.result_dir,
        args.conv_mode,
        args.debug_mode
    )
