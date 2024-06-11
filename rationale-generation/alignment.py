import os
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    last_inst_index = sentence.rfind('[/INST]')
    if last_inst_index != -1:
        return sentence[last_inst_index + len('[/INST]'):].strip()
    else:
        return sentence



def create_few_shot_prompt_4_shots(text, label, caption, web_entities):
    label = "Not Hateful." if label == "not_hateful" else "Hateful."
        
    messages = [
        {"role": "user", 
            "content": f"""## Definition of Hate Speech
            Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

            ## Instruction
            Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful". Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm. 
            
            ## Demonstration Examples
            ### Example 1
            Meme Text Overlay: White men are fucking trash. https://t.co/0NSJlGfHre 
            Meme Caption: a green trash can with a pile of wood in front of a brick wall. 
            Meme Web Entities: Wall, Image, Graphics, Illustration, Photography, Photograph, Graphic design, Vector graphics, Brickwork, Stock illustration
            Answer: Hateful
            Rationale: Targeted Group: The meme targets white men, singling them out based on their race and gender, which can be seen as discriminatory.\n\nDerogatory Imagery or Language: The imagery of a green trash can with a pile of wood in front of a brick wall, juxtaposed with the text 'White men are fucking trash,' reinforces derogatory imagery by associating white men with literal garbage, further degrading and dehumanizing them.\n\n Impact on Bias/Stereotypes: By categorizing an entire group as 'trash' and depicting them in a derogatory manner, the meme perpetuates negative biases and stereotypes, fostering division and prejudice rather than promoting understanding and unity. This can contribute to further polarization and discrimination within society.\n\n In summary, this meme is hateful.
            
            ### Example 2
            Meme Text Overlay: Real ass bitch give a fuck bout a nigga https://t.co/2M5VFMSC1d 
            Meme Caption: a woman in a store wearing sunglasses and a butterfly on her head
            Meme Web Entities: Isan Elba, other, Image, Photograph, monthly, available, Nudity, Zimcelebs, Nude, Pantyhose
            Answer: Not Hateful.
            Rationale: Targeted group: The meme does not directly target any specific group based on race, ethnicity or gender. \n Derogatory imagery/Language: While the term 'nigga' could be considered offensive in certain contexts, its use in this meme is not explicitly derogatory. It appears to be part of a colloquial expression rather than a deliberate slur. The imagery of a woman in a store wearing sunglasses and a butterfly on her head does not contribute to the derogatory nature of the meme. \n\n Impact on bias/stereotypes: There is no evident connection between the text and any perpetuation of bias, stereotypes, prejudice, discrimination, or inciting harm. \n\n In summary, this meme is not hateful.
            
            ### Example 3
            Meme Text Overlay: 9/10 if she can drive good, her pussy garbage. Or she a dyke https://t.co/NK2F20HtrG
            Meme Caption: a man sitting at a table with a sign that says change my mind
            Meme Web Entities: Joe Exotic, Meme, Tiger King, Imgur, Internet meme, Imgflip, Image, GIF, Demotivational Maker
            Answer: Hateful.
            Rationale: Targeted group: The meme targets women, particularly those who are capable drivers or lesbians, suggesting that their sexuality is tied to their driving skills. This reduces them to stereotypes and objectifies them based on arbitrary criteria. \n\n Derogatory Imagery/Language: The use of derogatory terms such as 'garbage' and 'dyke' to describe women who do not fit into the narrow, heteronormative standards perpetuates harmful stereotypes and reinforces discrimination against women and LGBTQ+ individuals. The imagery of a man sitting at a table with a sign that says change my mind suggests an open invitation to debate, but in reality, serves to reinforce and validate the hateful sentiments expressed in the meme. \n\n Impact on Bias/Stereotypes: By equating a woman's driving ability with her sexuality, the meme contributes to a culture of misogyny and homophobia. \n\n In summary, this meme is hateful.
            
            ### Example 4 
            Meme Text Overlay: You literally were cheating on your bf like 2 months ago with my person, you AINT loyal wtf you mean? Fucking cunt.
            Meme Caption: https://t.co/55xyQekcRD a twitter post with a picture of a sandwich and a picture of a person
            Meme Web Entities: Cartoon, Text, Screenshot, Meter
            Answer: Not Hateful
            Rationale: Targeted Group: The meme does not target a specific group, race, ethnicity, religion, or sexual orientation.  \n\n Derogatory Imagery/Language: While the language used in the text is offensive and derogatory towards the individual mentioned, it does not generalize or dehumanize an entire group of people. The derogatory terms used (such as 'cunt') are aimed at criticizing the behavior of the woman in the meme, rather than expressing hatred towards a particular group. The imagery of a sandwich and person does not convey hate or discrimination. \n\n Impact on Bias/Stereotypes: Although the language used in the meme is inappropriate, it does not appear to perpetuate bias or incite harm against any specific group. \n\n In summary, this meme is not hateful.
            
            ## Task
            Answer the following content and respond with either "Hateful" or "Not Hateful" based on the provided definition of hate speech. Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm. 

            Meme Text Overlay: {text}
            Meme Caption: {caption}
            Meme Web Entities: {web_entities}
            Answer: {label}
            Rationale:"""
        }
    ]

    return messages

def create_few_shot_prompt_10_shots(text, label, caption, web_entities):
    label = "Not Hateful." if label == "not_hateful" else "Hateful."
        
    messages = [
        {"role": "user", 
            "content": f"""## Definition of Hate Speech
            Hate speech is any communication that belittles, discriminates against, or incites violence against individuals or groups based on attributes such as race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics. This includes, but is not limited to, slurs, threats, dehumanizing language, and advocating for exclusion or violence against these individuals or groups.

            ## Instruction
            Evaluate the content using the definition of hate speech to determine if it is considered as hateful. Respond with either "Hateful" or "Not Hateful". Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm. 
            
            ## Demonstration Examples
            ### Example 1
            Meme Text Overlay: holy shit look at the gas bill
            Meme Caption: a woman with glasses looking at a piece of paper with a laptop in front of her
            Meme Web Entities: Image, Annie Spratt, stock.xchng, Photo agency, Money, Photograph, Debt, Paper, Finance
            Answer: Not Hateful.
            Rationale: Targeted Group: There is no targeted group in this meme as it is not based on race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics.\n\nDerogatory Imagery/Language: The imagery in the meme is a woman looking at a gas bill, which does not contain any derogatory imagery or language towards any group.\n\nImpact on Bias/Stereotypes: The meme does not contain any content that perpetuates bias, stereotypes, prejudice, discrimination, or incites harm towards any group. It is simply a humorous portrayal of the unexpected high cost of a gas bill.\n\nIn summary, the meme is not hateful.
            
            ### Example 2
            Meme Text Overlay: if this is what they call pride.. then i don't wanna see what they call shame
            Meme Caption: a man holding a rainbow flag
            Meme Web Entities: Lesbian and Gay City Festival, LGBT pride, Homosexuality, Sexual orientation, Germany, LGBT in Germany, Festival, LGBT community, Hatred
            Answer: Hateful.
            Rationale: Targeted Group: The meme targets the LGBTQ+ community, specifically those participating in LGBT pride events, implying that their pride is something to be ashamed of.\n\nDerogatory Imagery/Language: The use of the word \"shame\" in the meme text denotes a negative or derisive attitude towards the LGBTQ+ community, perpetuating discrimination based on their sexual orientation.\n\nImpact on Bias/Stereotypes: By implying that there is something shameful or wrong about pride in one's sexual orientation, the meme reinforces negative stereotypes and perpetuates biases, making it hateful.\n\nIn summary, the meme is hateful.

            ### Example 3 
            Meme Text Overlay: \"what part of \"i didn't find shit\" do you libtards not understand? \"
            Meme Caption: a close up of a person wearing a suit and tie
            Meme Web Entities: Thomas Matlack, Photograph, Image, Reddit, November, Main idea, Getty Images, r/malefashionadvice
            Answer: Not Hateful.
            Rationale: Targeted Group: The meme does not specifically target any particular group based on race, religion, ethnicity, gender, sexual orientation, disability, or other distinguishing characteristics.\n\nDerogatory Imagery/Language: The meme uses the term \"libtards,\" which some people may consider derogatory towards those holding liberal political views. However, this term is more of a political affiliation rather than a targeted attack on a specific group based on any of the aforementioned characteristics.\n\nImpact on Bias/Stereotypes: While the meme may not be agreeable to everyone, it does not contribute to perpetuating negative biases, stereotypes, prejudice, discrimination, or inciting harm against any specific group. Political disagreements are a common occurrence and do not inherently involve hate speech.\n\nIn summary, the meme is not hateful.
            
            ### Example 4
            Meme Text Overlay: sea monkeys a group of children running through the water on the beach
            Meme Caption: a group of children running through the water on the beach
            Meme Web Entities: Children's rights, Human rights, African Charter on the Rights and Welfare of the Child, Rights, African Union, African Charter on Human and Peoples' Rights, Organisation of African Youth, Convention on the Rights of the Child, African Committee of Experts on the Rights and Welfare of the Child
            Answer: Hateful.
            Rationale: Targeted Group: The term \"sea monkeys\" is often used as a derogatory term for black people, particularly in a historical context when black people were compared unfavorably to white people, with the implication being that black people were not human and more akin to aquatic creatures. The use of this term in the meme could be seen as perpetuating this harmful stereotype.\n\nDerogatory Imagery/Language: While the imagery of children running through the water on the beach is not inherently derogatory, the use of the term \"sea monkeys\" in relation to the children could be seen as dehumanizing and derogatory.\n\nImpact on Bias/Stereotypes: By perpetuating this harmful stereotype, the meme contributes to a culture of racism and discrimination, as it suggests that black people are not fully human and deserve to be treated as less than others. This is not only offensive but also contributes to the perpetuation of negative biases and stereotypes.\n\nIn summary, the meme is hateful.

            ### Example 5
            Meme Text Overlay: if you're proud to be white then this is the radio station for you
            Meme Caption: a group of men with white beards and black hats and sweatshirts hold signs that
            Meme Web Entities: White nationalism, White people, Counter-protest, White Lives Matter, White pride, Nationalism, Traditionalist Worker Party, Supremacism, Demonstration
            Answer: Not Hateful.
            Rationale: Targeted Group: The meme does not explicitly target any specific individual or group based on their race. Instead, it refers to \"white people\" in a general sense, which does not equate to discrimination or incitement towards a particular group.\n\nDerogatory Imagery/Language: The meme does not contain any derogatory imagery or language. The imagery of the men holding signs does not inherently convey hate or discrimination.\n\nImpact on Bias/Stereotypes: While the phrase \"proud to be white\" can be associated with white supremacist and white nationalist groups, it does not necessarily imply hate speech. In this meme, it seems more like a playful nod to a group of men proudly identifying as white. The absence of any content or language that perpetuates bias or stereotypes or incites harm against any particular group helps reinforce this interpretation.\n\nIn summary, the meme is not hateful.
            
            ### Example 6
            Meme Text Overlay: it's not a hijab it's a diaper, for peoples with shit for brains
            Meme Caption: a young woman with a white scarf on her head
            Meme Web Entities: Ilhan Omar, Image, United States, Photograph, United States Representative, Minnesota Democratic\u2013Farmer\u2013Labor Party, Muslim Collective, Internet meme, The Squad
            Answer: Hateful.
            Rationale: Targeted Group: The meme targets individuals who wear hijabs, which is a religious symbol for Muslims. By comparing the hijab to a diaper and implying that those who wear it have \"shit for brains,\" the meme is discriminatory towards Muslims and perpetuates negative stereotypes.\n\nDerogatory Imagery or Language: The text \"it's not a hijab it's a diaper\" mocks the religious beliefs of Muslims and demeans those who choose to wear the hijab. This language is derogatory and dehumanizing, reducing people who wear hijabs to the status of babies or individuals with inferior intellects.\n\nImpact on Bias/Stereotypes: By associating the hijab with fecal matter and implying that those who wear it are intellectually deficient, the meme contributes to a culture of bias and discrimination against Muslims. It reinforces negative stereotypes and perpetuates the idea that Muslims are inferior or less intelligent than other religious or ethnic groups. This can contribute to further marginalization and exclusion of Muslims within society.\n\nIn summary, the meme is hateful.

            ### Example 7
            Meme Text Overlay: slaves got whipped back in the day
            Meme Caption: a drawing of a group of people
            Meme Web Entities: Slavery, Territory of Orleans, Slave ship, United States, Abolitionism, Atlantic World, History, Hist\u00f3ria do racismo, Abolitionism in the United States
            Answer: Not Hateful.
            Rationale: Targeted Group: The meme does not directly target any specific group based on race, ethnicity, or nationality as the text and image refer to the historical practice of slavery which affected various groups of people around the world.\n\nDerogatory Imagery/Language: The imagery used in the meme is a drawing of a group of people, which does not contain any overtly hateful or derogatory imagery or language.\n\nImpact on Bias/Stereotypes: While the topic of slavery is inherently discriminatory and associated with immense suffering and oppression, the meme itself does not demonstrate any contemporary expressions of hate or racism towards any particular group. In fact, this meme may serve as a reminder of the historical injustices and the ongoing struggle for equality and human rights. Moreover, this meme does not suggest dehumanizing or derogatory stereotypes of any group of people.\n\nIn summary, the meme is not hateful.
            
            ### Example 8
            Meme Text Overlay: your deportation bus has arrived
            Meme Caption: a picture of a bulldozer
            Meme Web Entities: Bulldozer, Construction, Excavator, Loader, Heavy equipment, New construction, Grader, Backhoe loader, stock.xchng, Topadora
            Answer: Hateful.
            Rationale: Targeted Group: The meme alludes to deportation, which is a sensitive issue and often used in a hateful and discriminatory manner towards immigrants, specifically those who are undocumented. By referencing a \"deportation bus,\" the meme implicitly targets this group based on their immigrant status.\n\nDerogatory Imagery/Language: The imagery of a bulldozer further reinforces the derogatory nature of the meme. Bulldozers are often used in a destructive context, symbolizing the forceful removal of people or things. Associating the term \"deportation\" with the destructive imagery of a bulldozer perpetuates negative sentiments and reinforces harmful stereotypes about immigrants.\n\nImpact on bias/stereotypes: By using the words \"deportation bus\" and the imagery of a bulldozer, the meme contributes to a culture of fear, prejudice, and discrimination towards immigrants. It propagates the harmful stereotype that immigrants are a problem or a burden to society and should be forcibly removed. This can have a lasting impact on the way people perceive immigrants and contribute to a hostile environment.\n\nIn summary, the meme is hateful.

            ### Example 9
            Meme Text Overlay: they will have a surgery soon
            Meme Caption: a woman in a white hijab looks at the camera
            Meme Web Entities: Laleh Bijani, Ladan Bijani, Conjoined twins, Twin, Ladan and Laleh Bijani, Abigail and Brittany Hensel, The Rezakhanovy sisters, Getty Images, Getty Images
            Answer: Not Hateful.
            Rationale: Targeted Group: The meme does not specifically target a particular race, religion, or ethnicity. The woman in the meme is wearing a hijab, which is a religious and cultural head covering worn by some Muslim women. However, the meme does not use derogatory language or imagery to target Muslims or any other group.\n\nDerogatory Imagery or Language: The meme does not contain any derogatory imagery or language that could be considered hateful or discriminatory towards Muslims or any other group. The Text Overlay \"they will have a surgery soon\" is not derogatory and does not perpetuate any negative stereotypes or biases.\n\nImpact on Bias/Stereotypes: The meme does not contribute to perpetuating any negative bias, stereotypes, prejudice, discrimination, or inciting harm towards any specific group. It is simply reporting a fact, that the women in the meme are conjoined twins and will undergo surgery.\n\nIn summary, the meme is not hateful.

            ### Example 10
            Meme Text Overlay: i got 99 problems and chromosomes are one
            Meme Caption: a man holding two dumbbells in his hands
            Meme Web Entities: Down syndrome, Syndrome, Physical therapy, Gym, National Down Syndrome Society, Exercise, Stretching, Muscle, Sitting, Teenage girl
            Answer: Hateful.
            Rationale: Targeted Group: The meme implies that having chromosomes associated with Down syndrome is a problem. This singling out and making light of a medical condition is discriminatory.\n\nDerogatory Imagery/Language: Using the term \"chromosomes\" to refer to individuals with Down syndrome reinforces the outdated and harmful belief that people with this condition are defined solely by their genetic makeup. This language perpetuates negative stereotypes and contributes to the stigmatization of those with Down syndrome.\n\nImpact on Bias/Stereotypes: By perpetuating stereotypes and minimizing individuals with Down syndrome as mere \"problems,\" the meme reinforces negative biases and reinforces a lack of understanding and inclusion for the community. Additionally, the use of this language can serve to justify exclusion, bullying, and marginalization towards those with Down syndrome.\n\nIn summary, the meme is hateful.

            ## Task
            Answer the following content and respond with either "Hateful" or "Not Hateful" based on the provided definition of hate speech. Briefly provide an explanation, in no more than three points, for the post being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm. 

            Meme Text Overlay: {text}
            Meme Caption: {caption}
            Meme Web Entities: {web_entities}
            Answer: {label}
            Rationale:"""
        }
    ]

    return messages

# def create_few_shot_prompt_4_shots(text, label, caption, webentities):
#     hateful = None
#     if label == "not_hateful":
#         hateful = "not hateful"
#     else:
#         hateful = "hateful"
        
#     messages = [
#         {"role": "user", "content": f"Determine whether the following meme is hateful. \nMeme Text: White men are fucking trash. https://t.co/0NSJlGfHre \n Meme Caption: a green trash can with a pile of wood in front of a brick wall. \n Web Entities: Wall, Image, Graphics, Illustration, Photography, Photograph, Graphic design, Vector graphics, Brickwork, Stock illustration"},
#         {"role": "assistant", "content": f"Answer: hateful"},
#         {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
#         {"role": "assistant", "content": "Answer: Targeted Group: The meme targets white men, singling them out based on their race and gender, which can be seen as discriminatory.\n\nDerogatory Imagery or Language: The imagery of a green trash can with a pile of wood in front of a brick wall, juxtaposed with the text 'White men are fucking trash,' reinforces derogatory imagery by associating white men with literal garbage, further degrading and dehumanizing them.\n\n Impact on Bias/Stereotypes: By categorizing an entire group as 'trash' and depicting them in a derogatory manner, the meme perpetuates negative biases and stereotypes, fostering division and prejudice rather than promoting understanding and unity. This can contribute to further polarization and discrimination within society.\n\n In summary, this meme is hateful."},

#         {"role": "user", "content": f"Determine whether the following meme is hateful. \nMeme Text: Real ass bitch give a fuck bout a nigga https://t.co/2M5VFMSC1d \n Meme Caption: a woman in a store wearing sunglasses and a butterfly on her head \n Web Entities: Isan Elba, other, Image, Photograph, monthly, available, Nudity, Zimcelebs, Nude, Pantyhose"},
#         {"role": "assistant", "content": f"Answer: not hateful"},
#         {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
#         {"role": "assistant", "content": "Answer: Targeted group: The meme does not directly target any specific group based on race, ethnicity or gender. \n Derogatory imagery/Language: While the term 'nigga' could be considered offensive in certain contexts, its use in this meme is not explicitly derogatory. It appears to be part of a colloquial expression rather than a deliberate slur. The imagery of a woman in a store wearing sunglasses and a butterfly on her head does not contribute to the derogatory nature of the meme. \n\n Impact on bias/stereotypes: There is no evident connection between the text and any perpetuation of bias, stereotypes, prejudice, discrimination, or inciting harm. \n\n In summary, this meme is not hateful."},
        
#         {"role": "user", "content": f"Determine whether the following meme is hateful. \nMeme Text: 9/10 if she can drive good, her pussy garbage. Or she a dyke https://t.co/NK2F20HtrG \n Meme Caption: a man sitting at a table with a sign that says change my mind \n Web Entities: Joe Exotic, Meme, Tiger King, Imgur, Internet meme, Imgflip, Image, GIF, Demotivational Maker"},
#         {"role": "assistant", "content": f"Answer: hateful"},
#         {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
#         {"role": "assistant", "content": "Answer: Targeted group: The meme targets women, particularly those who are capable drivers or lesbians, suggesting that their sexuality is tied to their driving skills. This reduces them to stereotypes and objectifies them based on arbitrary criteria. \n\n Derogatory Imagery/Language: The use of derogatory terms such as 'garbage' and 'dyke' to describe women who do not fit into the narrow, heteronormative standards perpetuates harmful stereotypes and reinforces discrimination against women and LGBTQ+ individuals. The imagery of a man sitting at a table with a sign that says change my mind suggests an open invitation to debate, but in reality, serves to reinforce and validate the hateful sentiments expressed in the meme. \n\n Impact on Bias/Stereotypes: By equating a woman's driving ability with her sexuality, the meme contributes to a culture of misogyny and homophobia. \n\n In summary, this meme is hateful. "},
        
#         {"role": "user", "content": f"Determine whether the following meme is hateful. \nMeme Text: You literally were cheating on your bf like 2 months ago with my person, you AINT loyal wtf you mean? Fucking cunt. \n Meme Caption: https://t.co/55xyQekcRD a twitter post with a picture of a sandwich and a picture of a person \n Web Entities: Cartoon, Text, Screenshot, Meter"},
#         {"role": "assistant", "content": f"Answer: not hateful"},
#         {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as not hateful. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
#         {"role": "assistant", "content": "Answer: Targeted Group: The meme does not target a specific group, race, ethnicity, religion, or sexual orientation.  \n\n Derogatory Imagery/Language: While the language used in the text is offensive and derogatory towards the individual mentioned, it does not generalize or dehumanize an entire group of people. The derogatory terms used (such as 'cunt') are aimed at criticizing the behavior of the woman in the meme, rather than expressing hatred towards a particular group. The imagery of a sandwich and person does not convey hate or discrimination. \n\n Impact on Bias/Stereotypes: Although the language used in the meme is inappropriate, it does not appear to perpetuate bias or incite harm against any specific group. \n\n In summary, this meme is not hateful."},

#         {"role": "user", "content": f"Determine whether the following meme is hateful. Text: {text}. \n Caption: {caption}. \n Web Entities: {webentities}"},
#         {"role": "assistant", "content": f"Answer: {hateful}"},
#         {"role": "user", "content": f"Briefly provide an explanation, in no more than three points, for the meme being perceived as {hateful}. Your explanation should address the targeted group, any derogatory imagery or language used, and the impact it has on perpetuating bias, stereotypes, prejudice, discrimination or inciting harm."},
#         {"role": "assistant", "content": "Answer: "},
#     ]
#     return messages

import pandas as pd
def load_annotations(annotation_filepath):
    df = pd.read_json(annotation_filepath, lines=True)
    return df

def load_captions(caption_dir, filename):
    filepath = os.path.join(caption_dir, filename)
    with open(filepath) as f:
        d = json.load(f)

    return d['caption']

def load_entities(entity_dir, filename):
    filepath = os.path.join(entity_dir, filename)
    with open(filepath) as f:
        d = json.load(f)


    if 'webEntities' not in d['webDetection']:
        return "N/A"

    
    return ', '.join([x['description'] for x in d['webDetection']['webEntities'] if 'description' in x])


def main(annotations_file, img_dir, captions_dir, web_entities_dir, output_dir, prompt_approach, num_splits, split):
    # load all annotations
    df = load_annotations(annotations_file)
    
    output_dir = os.path.join(output_dir, prompt_approach)
    os.makedirs(output_dir, exist_ok=True)

    df = df[~df['id'].apply(lambda x: os.path.exists(os.path.join(output_dir, f"{x:05}.json")))]

    import math
    records_per_split = math.ceil(len(df) / num_splits)
    start = split * records_per_split
    end = (split + 1) * records_per_split
    df = df.iloc[start:end]

    device = "cuda" # the device to load the model onto
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    if prompt_approach == "few_shot_4_shots":
        create_prompt = create_few_shot_prompt_4_shots
    elif prompt_approach == "few_shot_10_shots":
        create_prompt = create_few_shot_prompt_10_shots
    else:
        raise NotImplementedError(f"prompt_approach `{prompt_approach}` not implemented")

    output_dir = os.path.join(output_dir, prompt_approach)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    import tqdm
    for memeID, text, label, gold_pc in tqdm.tqdm(zip(df['id'], df['text'], df['gold_hate'], df['gold_pc'])):
        label = label[0]

        filepath = os.path.join(output_dir, f"{memeID:05}.json")
        if os.path.exists(filepath):
            with open(filepath) as f:
                obj = json.load(f)
        else: 
            caption = load_captions(captions_dir, f"{memeID:05}.json")
            web_entities = load_entities(web_entities_dir, f"{memeID:05}.png")

            prompt = create_prompt(text, label, caption, web_entities)
            encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(device)

            model_inputs = encodeds
            model.to(device)

            generated_ids = model.generate(model_inputs, max_new_tokens=1028, do_sample=True)
            decoded = tokenizer.batch_decode(generated_ids)

            output = decoded[0]
            cleaned_explanation = remove_prompt_from_output(output)

            obj = {
                "id": f"{memeID:05}",
                "img": os.path.join(img_dir, f"{memeID:05}.png"),
                "caption": caption,
                "web_entities": web_entities,
                "text": text,
                "content": text + " " + caption,
                "label": label,
                "gold_pc": gold_pc,
                "rationale": cleaned_explanation
            }

            print(filepath)
            with open(filepath, "w+") as f:
                json.dump(obj, f)

        records.append(obj)

    if num_splits == 1:
        records = []
        for record_id in df['id']:
            data_filepath = os.path.join(output_dir, f"{record_id:05}.json")
            with open(data_filepath) as f:
                obj = json.load(f)
                obj['ID'] = record_id
                records.append(obj)

        data_filepath = os.path.join(output_dir, f"compiled.jsonl")
        with open(data_filepath, "w+") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

import argparse
parser = argparse.ArgumentParser("temporary for file splitting")
parser.add_argument("--annotation_filepath", type=str)
parser.add_argument("--img_dir", type=str)
parser.add_argument("--captions_dir", type=str)
parser.add_argument("--web_entities_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--prompt_approach", type=str, choices=["few_shot_4_shots", "few_shot_10_shots"])
parser.add_argument("--num_splits", type=int, required=True)
parser.add_argument("--split", type=int, required=True)
args = parser.parse_args()

main(args.annotation_filepath, args.img_dir, args.captions_dir, args.web_entities_dir, args.output_dir, args.prompt_approach, args.num_splits, args.split)