import re
import tqdm
import os
import torch
import json
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from utils import load_inference_dataset, load_support_dataset, load_images

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

from fs_utils import prepare_inputs_llava

# Reference: https://github.dev/haotian-liu/LLaVA/pull/1502/files

def replace_image_tokens(qs, mm_use_im_start_end):
    # qs = qs_orig.copy()
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
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

def main(
        model_path, 
        model_base, 
        annotation_filepath, 
        caption_dir, 
        image_dir, 
        result_dir, 
        conv_mode, 
        prompt_format,
        use_demonstration,
        demonstration_selection,
        demonstration_distribution,
        support_filepaths,
        support_caption_dirs,
        support_feature_dirs,
        support_image_dirs,
        sim_matrix_filepath,
        debug_mode,
        shots
    ):
    os.makedirs(result_dir, exist_ok=True)

    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    # Data
    inference_annots = load_inference_dataset(annotation_filepath, caption_dir, None, image_dir)
    support_annots = []
    for filepath, support_caption_dir, support_feature_dir, support_image_dir in zip(
        support_filepaths, 
        support_caption_dirs, 
        support_feature_dirs,
        support_image_dirs,
    ):
        annots = load_support_dataset(filepath, support_caption_dir, support_feature_dir, support_image_dir)
        support_annots += annots
    
    with open(sim_matrix_filepath, 'rb') as f:
        sim_matrix = np.load(f)
        labels = np.load(f)

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
        else:
            # Prepare Conv
            qs, image_paths = prepare_inputs_llava(
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

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Parse Images
            images = load_images(image_paths)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            )
            images_tensor = [x.to(model.device, dtype=torch.float16) for x in images_tensor]

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

    parser.add_argument("--use_demonstrations", action="store_true")
    parser.add_argument("--prompt_format", choices=["system_prompt", "single_prompt", "multi_turn_prompt"])
    parser.add_argument("--demonstration_selection", choices=["random", "tf-idf", "bm-25"])
    parser.add_argument("--demonstration_distribution", choices=["equal", "top-k"])
    parser.add_argument("--support_filepaths", nargs='+')
    parser.add_argument("--support_caption_dirs", nargs='+')
    parser.add_argument("--support_feature_dirs", nargs='+')
    parser.add_argument("--support_image_dirs", nargs='+')
    parser.add_argument("--sim_matrix_filepath", type=str, required=True)
    parser.add_argument("--shots", type=int, required=True)
    
    args = parser.parse_args()

    main(
        args.model_path,
        args.model_base,
        args.annotation_filepath,
        args.caption_dir,
        args.image_dir,
        args.result_dir,
        args.conv_mode,
        args.prompt_format,
        args.use_demonstrations,
        args.demonstration_selection,
        args.demonstration_distribution,
        args.support_filepaths,
        args.support_caption_dirs,
        args.support_feature_dirs,
        args.support_image_dirs,
        args.sim_matrix_filepath,
        args.debug_mode,
        args.shots
    )
