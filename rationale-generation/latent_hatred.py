from transformers import AutoModelForCausalLM, AutoTokenizer

# load models
device = "cuda" # the device to load the model onto
import os
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

#util scripts
import json
def append_dict_to_jsonl(dictionary, filename):
    with open(filename, 'a') as file:
        json.dump(dictionary, file)
        file.write('\n')

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
        {"role": "user", "content": f"Determine whether the following post is hateful. Text: radical radical radical radical radical radical muslims cheer terror bombings in Boston."},
        {"role": "assistant", "content": f"Answer: hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted Group: The post targets Muslims, specifically those labeled as 'radical', implying that all Muslims are inherently linked to terrorism. This generalization unfairly associates an entire religious group with extremist actions.\n\n Derogatory Imagery/Language: The repetitive use of the word 'radical' alongside 'terror bombings' conjures up a highly negative and alarming image, reinforcing the stereotype that all Muslims are violent extremists. \n\n Impact on Bias/Stereotypes: By framing Muslims as supporters of terrorism, the post fuels existing biases and prejudices against the Muslim community. This kind of rhetoric can lead to real-world harm, such as hate crimes and discrimination, by fostering fear and mistrust towards Muslims. It also serves to further polarize communities and incite division rather than promoting understanding and dialogue. \n\n In summary, this post is hateful."},

        {"role": "user", "content": f"Determine whether the following post is hateful. Text: the white mans 200 year run is over . i hope you took advantage of your privilege while yo."},
        {"role": "assistant", "content": f"Answer: not hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted Group: The post targets white individuals as a collective group, suggesting that their historical dominance or privilege is coming to an end. \n\n Derogatory Imagery/Language: While the language used in the post may be seen as blunt or provocative, it does not contain overtly derogatory terms or explicit hate speech against white individuals.\n\n Impact on Bias/Stereotypes: The post perpetuates a narrative of power dynamics shifting away from historically dominant groups, which could either challenge or reinforce existing biases and stereotypes. It may foster resentment or defensiveness among some individuals, while others may interpret it as a call for accountability and equity. \n\n In summary, this post is not hateful."},
        
        {"role": "user", "content": f"Determine whether the following post is hateful. Text: #jews & #niggers destroy & pervert everything they touch #jewfail #niggerfail"},
        {"role": "assistant", "content": f"Answer: hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted Group: The post uses derogatory terms 'jews' and 'niggers' to target Jewish and Black people, respectively. These terms have historically been used as racial slurs to demean and dehumanize these groups. \n\n Derogatory Imagery/Language: The use of terms like 'destroy & pervert' conveys a negative and harmful stereotype about Jewish and Black people, suggesting that they inherently bring destruction and corruption wherever they go. Such language reinforces harmful stereotypes and perpetuates prejudice against these groups. \n\n  Impact on Bias/Stereotypes: By spreading messages that vilify and dehumanize specific Jews and Black people, this post contributes to the normalization of discrimination and hatred. \n\n In summary, this post is hateful."},
        
        {"role": "user", "content": f"Determine whether the following post is hateful. Text: the white mans 200 year run is over . i hope you took advantage of your privilege while yo."},
        {"role": "assistant", "content": f"Answer: not hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the post being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted Group: The post does not target a specific ethnic or religious group for discrimination or harm. While it mentions 'brown British' and 'Congress', it doesn't single out individuals based on their race, ethnicity, or religion. \n\n Derogatory Imagery or Language: While the use of hashtags like '#congressbrownbritish' and '#brownbritishcong' might suggest that the members of the Congress party are remnants of British colonial India, the post itself doesn't contain explicit derogatory language or imagery directed at any group. \n\n Impact on Bias/Stereotypes: The post expresses a political opinion rather than promoting hate speech. While it might perpetuate biases against members of parties like Congress, it doesn't incite direct harm or discrimination against any specific group based on their race, ethnicity, or religion. \n\n In summary, this post is not hateful."},

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
def main(annotations_file, output_file):
    # load all annotations
    data = []
    with open(annotations_file, 'r') as file:
        for line in file:
            # Load each line as JSON
            json_data = json.loads(line)
            data.append(json_data)


    for index, record in enumerate(data):
        prompt = create_prompt(record["post"], record["class"])
        encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)

        model_inputs = encodeds
        model.to(device)

        generated_ids = model.generate(model_inputs, max_new_tokens=340, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)

        output = decoded[0]
        cleaned_explanation = remove_prompt_from_output(output)
        record["mistral_instruct_statement"] = cleaned_explanation
        append_dict_to_jsonl(record, output_file)
        
annotations_file = '/mnt/data1/datasets/hatespeech/latent_hatred/truncated/train.jsonl'
output_file = 'latent-hatred-train-output.jsonl'
main(annotations_file, output_file)