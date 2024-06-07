from transformers import AutoModelForCausalLM, AutoTokenizer

# load models
device = "cuda" # the device to load the model onto
import os
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

#util scripts
import json
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def remove_prompt_from_output(sentence):
    last_inst_index = sentence.rfind('Answer:')
    if last_inst_index != -1:
        return sentence[last_inst_index + len('Answer:'):].strip()
    else:
        return sentence

def create_prompt(text, label):
    hateful = None
    if label == 0:
        hateful = "not hateful"
    else:
        hateful = "hateful"
        
    messages = [
        {"role": "user", "content": f"Determine whether the following post is hateful. Text: u lucky  u don't live in pakistan  muslims will kill u 4 just being a jew ."},
        {"role": "assistant", "content": f"Answer: hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer:  Targeted Group: The post targets Muslims by suggesting they are inherently violent towards Jews, promoting a negative stereotype. \n\n  Derogatory imagery/Language: The statement uses threatening language, implying that Muslims will commit violence against Jews, which is derogatory and inflammatory. \n\n  Impact on bias/stereotypes: This post reinforces harmful stereotypes and prejudices against Muslims, and could incite fear and hostility towards them."},

        {"role": "user", "content": f"Determine whether the following post is hateful. Text: yes ... you do hate blacks unless they accept white culture or white supremacists"},
        {"role": "assistant", "content": f"Answer: not hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer:  Targeted Group: The post does not target a specific racial or ethnic group with harmful intent; rather, it comments on the behavior or attitudes of some people towards black individuals.\n\n  Derogatory Imagery/Language: There is no use of derogatory language aimed at a particular group. The statement critiques a perceived behavior rather than attacking any group with harmful labels. \n\n  Impact on Bias/Stereotypes: While the post discusses sensitive racial issues, it does not perpetuate harmful stereotypes or incite prejudice or violence against any group. It highlights a societal issue without encouraging discrimination or harm."},
        
        {"role": "user", "content": f"Determine whether the following post is hateful. Text: my post on why gay people need to concede marriage = = >"},
        {"role": "assistant", "content": f"Answer: hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer:  Targeted Group: The post targets gay people by suggesting that they should concede their right to marriage, undermining their equality and rights. \n\n  Derogatory Imagery/Language: While the language is not explicitly derogatory, the implication that gay people should give up marriage rights is dismissive and disrespectful. \n\n  Impact on Bias/Stereotypes: The post perpetuates prejudice against gay people by advocating for the denial of their rights, reinforcing discriminatory attitudes and potentially inciting further discrimination and harm against the LGBTQ+ community."},
        
        {"role": "user", "content": f"Determine whether the following post is hateful. Text: hindus  sikhs in afghanistan feel neglected  discriminated - #hdl"},
        {"role": "assistant", "content": f"Answer: not hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer:  Targeted Group: The post discusses the experiences of Hindus and Sikhs in Afghanistan, but it does not target these groups with harmful intent. \n\n  Derogatory Imagery/Language: There is no derogatory language or imagery used in the post; it describes a situation of discrimination and neglect that these groups face. \n\n  Impact on Bias/Stereotypes: The post does not perpetuate any bias or stereotypes. Instead, ithighlights issues of discrimination and neglect, which can raise awareness and encourage support for these communities."},

        {"role": "user", "content": f"Determine whether the following post is hateful. Text: {text}."},
        {"role": "assistant", "content": f"Answer: {hateful}"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as {hateful}. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: "},
    ]
    
    return messages


# train annotations - /mnt/data1/datasets/hatespeech/latent_hatred/truncated/train.jsonl
# val annotations - /mnt/data1/datasets/hatespeech/latent_hatred/truncated/val.jsonl
# test annotations - /mnt/data1/datasets/hatespeech/latent_hatred/truncated/test.jsonl

import json
def main(annotations_file, output_dir, num_splits, split):
    # load all annotations
    data = []
    with open(annotations_file, 'r') as file:
        for line in file:
            # Load each line as JSON
            record = json.loads(line)

            # Remove records with existing rationales
            output_filepath = os.path.join(output_dir, f"{record['ID']}.json")
            if os.path.exists(output_filepath):
                continue
            
            data.append(record)

    print("Number Records Remaining:", len(data))
    import math
    records_per_split = math.ceil(len(data) / num_splits)
    start = split * records_per_split
    end = (split + 1) * records_per_split
    data = data[start:end]
    print("Split:", f"{split}/{num_splits}")
    print("Number Records Processing:", len(data))

    for index, record in enumerate(data):
        print(record["ID"])
        prompt = create_prompt(record["post"], record["class"])
        encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)

        model_inputs = encodeds
        model.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=340, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)

        output = decoded[0]
        cleaned_explanation = remove_prompt_from_output(output)

        output_filepath = os.path.join(output_dir, f"{record['ID']}.json")
        with open(output_filepath, "w+") as f:
            json.dump({"rationale": cleaned_explanation}, f)
        

annotations_file = '/mnt/data1/datasets/hatespeech/latent_hatred/annotations/explanations/stage1.jsonl'
output_dir = "/mnt/data1/datasets/hatespeech/latent_hatred/rationales/mistral-v0.2-7b/"

import argparse
parser = argparse.ArgumentParser("temporary for file splitting")
parser.add_argument("--num_splits", type=int, required=True)
parser.add_argument("--split", type=int, required=True)
args = parser.parse_args()

main(annotations_file, output_dir, args.num_splits, args.split)
