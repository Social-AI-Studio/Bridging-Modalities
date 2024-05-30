from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto
import os
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

import json
def append_dict_to_jsonl(dictionary, filename):
    with open(filename, 'a') as file:
        json.dump(dictionary, file)
        file.write('\n')

def combine_captions(directory):
    combined_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                img_name = os.path.splitext(filename)[0]
                caption = data["caption"]
                combined_dict[img_name] = caption
    return combined_dict

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def create_prompt(text, label, caption, webentities):
    hateful = None
    if label == "not_hateful":
        hateful = "not hateful"
    else:
        hateful = "hateful"
        
    messages = [
        {"role": "user", "content": f"Determine whether the following meme is hateful. Text: White men are fucking trash. https://t.co/0NSJlGfHre \n\n Caption: a green trash can with a pile of wood in front of a brick wall. \n\n Web Entities: Wall, Image, Graphics, Illustration, Photography, Photograph, Graphic design, Vector graphics, Brickwork, Stock illustration"},
        {"role": "assistant", "content": f"Answer: hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted Group: The meme targets white men, singling them out based on their race and gender, which can be seen as discriminatory.\n\nDerogatory Imagery or Language: The imagery of a green trash can with a pile of wood in front of a brick wall, juxtaposed with the text 'White men are fucking trash,' reinforces derogatory imagery by associating white men with literal garbage, further degrading and dehumanizing them.\n\n Impact on Bias/Stereotypes: By categorizing an entire group as 'trash' and depicting them in a derogatory manner, the meme perpetuates negative biases and stereotypes, fostering division and prejudice rather than promoting understanding and unity. This can contribute to further polarization and discrimination within society.\n\n In summary, this meme is hateful."},

        {"role": "user", "content": f"Determine whether the following meme is hateful. Text: Real ass bitch give a fuck bout a nigga https://t.co/2M5VFMSC1d \n\n Caption: a woman in a store wearing sunglasses and a butterfly on her head \n\n Web Entities: Isan Elba, other, Image, Photograph, monthly, available, Nudity, Zimcelebs, Nude, Pantyhose"},
        {"role": "assistant", "content": f"Answer: not hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted group: The meme does not directly target any specific group based on race, ethnicity or gender. \n\n Derogatory imagery/Language: While the term 'nigga' could be considered offensive in certain contexts, its use in this meme is not explicitly derogatory. It appears to be part of a colloquial expression rather than a deliberate slur. The imagery of a woman in a store wearing sunglasses and a butterfly on her head does not contribute to the derogatory nature of the meme. \n\n Impact on bias/stereotypes: There is no evident connection between the text and any perpetuation of bias, stereotypes, prejudice, discrimination, or inciting harm. \n\n In summary, this meme is not hateful."},
        
        {"role": "user", "content": f"Determine whether the following meme is hateful. Text: 9/10 if she can drive good, her pussy garbage. Or she a dyke https://t.co/NK2F20HtrG \n\n Caption: a man sitting at a table with a sign that says change my mind \n\n Web Entities: Joe Exotic, Meme, Tiger King, Imgur, Internet meme, Imgflip, Image, GIF, Demotivational Maker"},
        {"role": "assistant", "content": f"Answer: hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted group: The meme targets women, particularly those who are capable drivers or lesbians, suggesting that their sexuality is tied to their driving skills. This reduces them to stereotypes and objectifies them based on arbitrary criteria. \n\n Derogatory Imagery/Language: The use of derogatory terms such as 'garbage' and 'dyke' to describe women who do not fit into the narrow, heteronormative standards perpetuates harmful stereotypes and reinforces discrimination against women and LGBTQ+ individuals. The imagery of a man sitting at a table with a sign that says change my mind suggests an open invitation to debate, but in reality, serves to reinforce and validate the hateful sentiments expressed in the meme. \n\n Impact on Bias/Stereotypes: By equating a woman's driving ability with her sexuality, the meme contributes to a culture of misogyny and homophobia. \n\n In summary, this meme is hateful. "},
        
        {"role": "user", "content": f"Determine whether the following meme is hateful. Text: You literally were cheating on your bf like 2 months ago with my person, you AINT loyal wtf you mean? Fucking cunt. \n\n Caption: https://t.co/55xyQekcRD a twitter post with a picture of a sandwich and a picture of a person \n\n Web Entities: Cartoon, Text, Screenshot, Meter"},
        {"role": "assistant", "content": f"Answer: not hateful"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: Targeted Group: The meme does not target a specific group, race, ethnicity, religion, or sexual orientation.  \n\n Derogatory Imagery/Language: While the language used in the text is offensive and derogatory towards the individual mentioned, it does not generalize or dehumanize an entire group of people. The derogatory terms used (such as 'cunt') are aimed at criticizing the behavior of the woman in the meme, rather than expressing hatred towards a particular group. The imagery of a sandwich and person does not convey hate or discrimination. \n\n Impact on Bias/Stereotypes: Although the language used in the meme is inappropriate, it does not appear to perpetuate bias or incite harm against any specific group. \n\n In summary, this meme is not hateful."},

        {"role": "user", "content": f"Determine whether the following meme is hateful. Text: {text}. \n\n Caption: {caption}. \n\n Web Entities: {webentities}"},
        {"role": "assistant", "content": f"Answer: {hateful}"},
        {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as {hateful}. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
        {"role": "assistant", "content": "Answer: "},
    ]
    return messages

import json

# load all annotations
data = []
with open('/mnt/data1/datasets/temp/MMHS150K/annotations/train_truncated.jsonl', 'r') as file:
    for line in file:
        # Load each line as JSON
        json_data = json.loads(line)
        data.append(json_data)

def main():
    # load all captions
    combined_dict = combine_captions('/mnt/data1/datasets/temp/MMHS150K/captions/deepfillv2/blip2-opt-6.7b-coco')

    # load all web entities
    file_path = '/mnt/data1/datasets/temp/MMHS150K/final_entities.json'
    combined_web = read_json_file(file_path)

    for index, record in enumerate(data):
        print(index)

        if record["id"] in combined_web:
            prompt = create_prompt(record["tweet_text"], record["label"], combined_dict[record["id"]], combined_web[record["id"]])
            encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)

            model_inputs = encodeds
            model.to(device)

            generated_ids = model.generate(model_inputs, max_new_tokens=340, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)
            record["mistral_instruct_statement"] = decoded[0]
            append_dict_to_jsonl(record, 'mmhs-output.jsonl')
        else: 
            continue

main()