from transformers import AutoModelForCausalLM, AutoTokenizer

# load models
device = "cuda" # the device to load the model onto
import os


#util scripts
import json
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def remove_prompt_from_output(sentence):
    last_inst_index = sentence.rfind('[/INST]')
    if last_inst_index != -1:
        return sentence[last_inst_index + len('[/INST]'):].strip()
    else:
        return sentence

def create_zero_shot_prompt(text, label):
    label = "Not Hateful." if label == "not_hate" else "Hateful."
        
    messages = [
        {"role": "user", 
            "content": f"""## Definition of Hate Speech
            Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

            ## Instruction
            Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful". Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm. 

            Text: {text}"
            Answer: {label}
            Rationale:"""
        }
    ]

    return messages

def create_few_shot_prompt_4_shots(text, label):
    label = "Not Hateful." if label == "not_hate" else "Hateful."
        
    messages = [
        {"role": "user", 
            "content": f"""## Definition of Hate Speech
            Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

            ## Instruction
            Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful."
            
            ## Demonstration Examples
            ### Example 1
            Text: u lucky  u don't live in pakistan  muslims will kill u 4 just being a jew .
            Answer: Hateful
            Rationale: Targeted Group: The post targets Muslims by suggesting they are inherently violent towards Jews, promoting a negative stereotype. \n\n  Derogatory imagery/Language: The statement uses threatening language, implying that Muslims will commit violence against Jews, which is derogatory and inflammatory. \n\n  Impact on bias/stereotypes: This post reinforces harmful stereotypes and prejudices against Muslims, and could incite fear and hostility towards them. \n\n In summary, this post is hateful.
            
            ### Example 2
            Text: yes ... you do hate blacks unless they accept white culture or white supremacists
            Answer: Not Hateful.
            Rationale: Targeted Group: The post does not target a specific racial or ethnic group with harmful intent; rather, it comments on the behavior or attitudes of some people towards black individuals.\n\n  Derogatory Imagery/Language: There is no use of derogatory language aimed at a particular group. The statement critiques a perceived behavior rather than attacking any group with harmful labels. \n\n  Impact on Bias/Stereotypes: While the post discusses sensitive racial issues, it does not perpetuate harmful stereotypes or incite prejudice or violence against any group. It highlights a societal issue without encouraging discrimination or harm. \n\n In summary, this post is not hateful.
            
            ### Example 3
            Text: my post on why gay people need to concede marriage = = >
            Answer: Hateful.
            Rationale: Targeted Group: The post targets gay people by suggesting that they should concede their right to marriage, undermining their equality and rights. \n\n  Derogatory Imagery/Language: While the language is not explicitly derogatory, the implication that gay people should give up marriage rights is dismissive and disrespectful. \n\n  Impact on Bias/Stereotypes: The post perpetuates prejudice against gay people by advocating for the denial of their rights, reinforcing discriminatory attitudes and potentially inciting further discrimination and harm against the LGBTQ+ community.  \n\n In summary, this post is hateful.
            
            ### Example 4 
            Content: hindus  sikhs in afghanistan feel neglected  discriminated - #hdl
            Answer: Not Hateful
            Rationale: Targeted Group: The post discusses the experiences of Hindus and Sikhs in Afghanistan, but it does not target these groups with harmful intent. \n\n  Derogatory Imagery/Language: There is no derogatory language or imagery used in the post; it describes a situation of discrimination and neglect that these groups face. \n\n  Impact on Bias/Stereotypes: The post does not perpetuate any bias or stereotypes. Instead, ithighlights issues of discrimination and neglect, which can raise awareness and encourage support for these communities.  \n\n In summary, this post is not hateful.
            
            ## Task
            Answer the following content and respond with either "Hateful" or "Not Hateful" based on the provided definition of hate speech.

            Content: {text}"
            Answer: {label}
            Rationale:"""
        }
    ]

    return messages

def create_few_shot_prompt_10_shots(text, label):
    label = "Not Hateful." if label == "not_hate" else "Hateful."
        
    messages = [
        {"role": "user", 
            "content": f"""## Definition of Hate Speech
            Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

            ## Instruction
            Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful". Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm. 
            
            ## Demonstration Examples
            ### Example 1
            Text: question for all: what is the legal difference between a man wearing a dress, and a transwoman?
            Answer: Hateful.
            Rationale: Targeted Group: The post targets transgender people by implying that their gender identity is not legitimate, as it equates their identity with a man wearing a dress.\n\nDerogatory Imagery/Language: The use of the phrase \"a man wearing a dress\" suggests a discriminatory and dismissive attitude towards transgender women, questioning the validity of their gender identity.\n\nImpact on Bias/Stereotypes: The post reinforces harmful and degrading stereotypes about transgender individuals, questioning their identity and legitimacy, and potentially inciting prejudice or discrimination against them.\n\nIn summary, this post is hateful.

            ### Example 2
            Text: yes ... you do hate blacks unless they accept white culture or white supremacists
            Answer: Not Hateful.
            Rationale: Targeted Group: The post does not target a specific racial or ethnic group with harmful intent; rather, it comments on the behavior or attitudes of some people towards black individuals.\n\n  Derogatory Imagery/Language: There is no use of derogatory language aimed at a particular group. The statement critiques a perceived behavior rather than attacking any group with harmful labels. \n\n  Impact on Bias/Stereotypes: While the post discusses sensitive racial issues, it does not perpetuate harmful stereotypes or incite prejudice or violence against any group. It highlights a societal issue without encouraging discrimination or harm. \n\n In summary, this post is not hateful.

            ### Example 3
            Text: russian tv report on france's white genocide by mass immigration via #jewishimmi ...
            Answer: Hateful.
            Rationale: Targeted Group: The post targets white individuals in France by suggesting that mass immigration is a form of \"white genocide.\" This implication promotes harmful stereotypes that ethnic or cultural diversity poses a threat to white people.\n\nDerogatory Imagery/Language: The statement uses the term \"white genocide,\" which is a derogatory and inflammatory phrase that perpetuates racism and xenophobia. Additionally, the use of \"#jewishimmi\" implies a connection between Jewish people and the immigrants in question, potentially further solidifying racist and anti-Semitic attitudes.\n\nImpact on Bias/Stereotypes: This post reinforces harmful stereotypes and prejudices against white people, immigrants, and potentially Jewish individuals. It could incite fear and hostility towards these groups while promoting discriminatory and divisive beliefs.\n\nIn summary, this post is hateful.

            ### Example 4 
            Text: hindus  sikhs in afghanistan feel neglected  discriminated - #hdl
            Answer: Not Hateful.
            Rationale: Targeted Group: The post discusses the experiences of Hindus and Sikhs in Afghanistan, but it does not target these groups with harmful intent. \n\n  Derogatory Imagery/Language: There is no derogatory language or imagery used in the post; it describes a situation of discrimination and neglect that these groups face. \n\n  Impact on Bias/Stereotypes: The post does not perpetuate any bias or stereotypes. Instead, ithighlights issues of discrimination and neglect, which can raise awareness and encourage support for these communities.  \n\n In summary, this post is not hateful.

            ### Example 5
            Text:worried about her headscarf? i would have torched it!
            Answer: Hateful.
            Rationale: Targeted Group: The post targets people who wear headscarves, particularly associated with Muslim women.\n\nDerogatory Imagery/Language: The statement uses threatening language, suggesting violent harm towards individuals who wear headscarves.\n\nImpact on Bias/Stereotypes: This statement reinforces harmful stereotypes and prejudices against individuals who wear headscarves, potentially inciting fear and hostility towards Muslim women.\n\nIn summary, this post is hateful.

            ### Example 5
            Text: i wonder how many of those shootings involved white cops or klansmen
            Answer: Not Hateful.
            Rationale: The post makes a general comment about a specific racial group (white cops or Klansmen) in relation to a sensitive issue (shootings).\n\nDerogatory Imagery/Language: There is no derogatory language or imagery used in the post, but the statement might be perceived as insensitive given the historical context of white supremacist groups like the Klan.\n\nImpact on Bias/Stereotypes: The post does not intend to perpetuate harmful stereotypes or incite prejudice or violence towards any group. Instead, it appears that the author intends to bring attention to the disproportionate representation of racial groups in a certain issue or event.\n\nIn summary, this post is not hateful.
            
            ### Example 7
            Text: yo  c'mon back you little illegal aliens . we still got lots of us-born suckers we can smash out of line to make room
            Answer: Hateful.
            Rationale: Targeted Group: The post targets individuals who are undocumented immigrants, using a derogatory term (\"illegal aliens\").\n\nDerogatory Imagery/Language: The statement uses dehumanizing language by referring to people as \"illegal aliens,\" and it promotes violence against undocumented immigrants with the phrase \"we still got lots of us-born suckers we can smash out of line.\"\n\nImpact on Bias/Stereotypes: The post reinforces harmful stereotypes and prejudices against undocumented immigrants, encouraging discrimination and violence against them. It also contributes to a hostile and exclusionary environment for these individuals.\n\nIn summary, this post is hateful.

            ### Example 8
            Text: bbc headline on isis suicide bombing in germany : syrian migrant dies in german blast #wakeupamerica
            Answer: Not Hateful.
            Rationale: Targeted Group: The post mentions Syrian migrants but does not target them with harmful intent. Instead, it refers to a specific incident regarding a Syrian migrant in Germany, which is related to news reporting.\n\nDerogatory Imagery/Language: There is no derogatory language or imagery used in the post; it describes a tragic event related to refugee migration.\n\nImpact on Bias/Stereotypes: While the post does discuss migration, it does not perpetuate harmful stereotypes or incite prejudice or violence against any group. The hashtag #wakeupamerica may be seen as politically charged, but it does not directly target any specific group.\n\nIn summary, this post is not hateful.

            ### Example 9
            Text: handicapped is not a gender, bathroom sign people!!!
            Answer: Hateful.
            Rationale: Targeted Group: The statement targets individuals with disabilities by using a derogatory term and causing harm by invalidating their experiences and rights.\n\nDerogatory Imagery/Language: Using the term \"handicapped\" is disrespectful and stigmatizing. The phrase \"bathroom sign people\" dismisses the need for accommodations and is used as a derogatory insult against disabled individuals.\n\nImpact on Bias/Stereotypes: This post perpetuates harmful stereotypes and prejudices against individuals with disabilities. It minimizes the importance of accommodations and contributes to an environment that doesn't respect or support the needs of this marginalized group.\n\nIn summary, this post is hateful.

            ### Example 10
            Text: science explicitly explains that race is about much more than just skin color . #altright
            Answer: Not Hateful.
            Rationale: The statement does not target a specific racial or ethnic group with harmful intent, but rather provides a perspective on race.\n\nDerogatory Imagery/Language: There is no use of derogatory language aimed at a particular group. The statement provides a perspective on race.\n\nImpact on Bias/Stereotypes: While the post discusses race, it does not reinforce harmful stereotypes or prejudices against any group. It promotes a nuanced understanding of race that goes beyond skin color. However, it is important to note that the hashtag #altright has been associated with white supremacist and racist groups.\n\nIn summary, this post is not hateful.
            
            ## Task
            Answer the following content and respond with either "Hateful" or "Not Hateful" based on the provided definition of hate speech. Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm. 

            Content: {text}"
            Answer: {label}
            Rationale:"""
        }
    ]

    return messages

import json
def main(annotations_file, output_dir, prompt_approach, num_splits, split):

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if prompt_approach == "zero_shot":
        create_prompt = create_zero_shot_prompt
    elif prompt_approach == "few_shot_10_shots":
        create_prompt = create_few_shot_prompt_10_shots
    else:
        raise NotImplementedError(f"prompt_approach `{prompt_approach}` not implemented")

    output_dir = os.path.join(output_dir, prompt_approach)
    os.makedirs(output_dir, exist_ok=True)

    # load all annotations
    orig_data = []
    with open(annotations_file, 'r') as file:
        for line in file:
            # Load each line as JSON
            record = json.loads(line)
            orig_data.append(record)

    data = []
    for record in orig_data:
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

        generated_ids = model.generate(
            model_inputs, 
            max_new_tokens=1024, 
            do_sample=False, 
            num_beams=1
        )
        decoded = tokenizer.batch_decode(generated_ids)

        output = decoded[0]
        cleaned_explanation = remove_prompt_from_output(output)

        output_filepath = os.path.join(output_dir, f"{record['ID']}.json")
        obj = {
            "text": record['post'],
            "class": record['class'],
            "target_categories": record['target_categories'],
            "rationale": cleaned_explanation
        }
        with open(output_filepath, "w+") as f:
            json.dump(obj, f)
        
    if num_splits == 1:
        records = []
        for record in orig_data:
            data_filepath = os.path.join(output_dir, f"{record['ID']}.json")
            with open(data_filepath) as f:
                obj = json.load(f)
                obj['ID'] = record['ID']
                records.append(obj)

        data_filepath = os.path.join(output_dir, f"compiled.jsonl")
        with open(data_filepath, "w+") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

    

import argparse
parser = argparse.ArgumentParser("temporary for file splitting")
parser.add_argument("--annotation_filepath", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--prompt_approach", type=str, choices=["zero_shot", "few_shot_10_shots"])
parser.add_argument("--num_splits", type=int, required=True)
parser.add_argument("--split", type=int, required=True)
args = parser.parse_args()

main(args.annotation_filepath, args.output_dir, args.prompt_approach, args.num_splits, args.split)

# To Do (Temporary)
# - Sort the records in the different broad target categories
# - Pick some out for demonstration purposes
# - Determine good demonstration purposes
# - Run few-shot demonstrations

# - API Key
# - Model (Gemini 1.5 Pro & )