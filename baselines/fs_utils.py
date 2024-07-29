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

from llava.constants import IMAGE_PLACEHOLDER

def prepare_inputs(content, content_idx, prompt_format, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    # if prompt_format == "system_prompt":
    #     return prepare_inputs_system(
    #         content,
    #         content_idx, 
    #         use_demonstrations,
    #         demonstration_selection,
    #         demonstration_distribution,
    #         support_annots,
    #         sim_matrix,
    #         labels,
    #         k
    #     )
    if prompt_format == "single_prompt":
        return prepare_inputs_single_prompt(
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
    # elif prompt_format == "multi_turn_prompt":
    #     return prepare_inputs_system_conversational(
    #         content,
    #         content_idx, 
    #         use_demonstrations,
    #         demonstration_selection,
    #         demonstration_distribution,
    #         support_annots,
    #         sim_matrix,
    #         labels,
    #         k
    #     )
    else:
        raise NotImplementedError(f"prompt format '{prompt_format}' not implemented.")

def prepare_inputs_llava(content, content_idx, prompt_format, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
    # if prompt_format == "system_prompt":
    #     return prepare_inputs_system(
    #         content,
    #         content_idx, 
    #         use_demonstrations,
    #         demonstration_selection,
    #         demonstration_distribution,
    #         support_annots,
    #         sim_matrix,
    #         labels,
    #         k
    #     )
    if prompt_format == "single_prompt":
        return prepare_inputs_single_prompt_llava(
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
    # elif prompt_format == "multi_turn_prompt":
    #     return prepare_inputs_system_conversational(
    #         content,
    #         content_idx, 
    #         use_demonstrations,
    #         demonstration_selection,
    #         demonstration_distribution,
    #         support_annots,
    #         sim_matrix,
    #         labels,
    #         k
    #     )
    else:
        raise NotImplementedError(f"prompt format '{prompt_format}' not implemented.")

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

        qs, image_paths = [], []
        qs.append(INTRODUCTION)
        if demonstration_distribution == "equal":
            pass
        else:
            for index, s in enumerate(samples):
                answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
                example = EXAMPLE_TEMPLATE_WITH_RATIONALE.format(index=index+1, content=s['content'], rationale=s['rationale'], answer=answer)
                qs.append(example)

    question = QUESTION_TEMPLATE.format(content=content['content'])
    qs.append(question)

    return "".join(qs)


def prepare_inputs_single_prompt_llava(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
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

        qs, image_paths = [], []
        qs.append(INTRODUCTION)
        if demonstration_distribution == "equal":
            pass
        else:
            for index, s in enumerate(samples):
                answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
                example = EXAMPLE_TEMPLATE_WITH_RATIONALE.format(index=index+1, content=s['content_llava'], rationale=s['rationale'], answer=answer)
                qs.append(example)

                if IMAGE_PLACEHOLDER in example:
                    image_paths.append(s['img_path'])

    question = QUESTION_TEMPLATE.format(content=content['content_llava'])
    qs.append(question)
    image_paths.append(content['img_path'])

    return "".join(qs), image_paths

# def prepare_inputs_system(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
#     if use_demonstrations:
#         if demonstration_selection == "tf-idf":
#             similar_entries = tfidf_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
#             similar_entry_indices = [entry[0] for entry in similar_entries]
#             samples = [support_annots[index] for index in similar_entry_indices]
#         else:
#             raise NotImplementedError(f"demonstration selection '{demonstration_selection}' not found")

#         msg = [INTRODUCTION_WITHOUT_INSTRUCTIONS]
#         if demonstration_distribution == "equal":
#             pass
#         else:
#             for index, s in enumerate(samples):
#                 answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
#                 example = EXAMPLE_TEMPLATE_WITH_RATIONALE.format(
#                     index=index+1, 
#                     content=s['content'], 
#                     rationale=s['rationale'], 
#                     answer=answer
#                 )
#                 msg.append(example)

#     question = QUESTION_TEMPLATE.format(content=content['content'])
#     msg.append(question)
    

#     joined_examples = "\n".join(msg)
#     prompt = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": joined_examples}
#     ]
#     print(json.dumps(prompt, indent=2))
#     exit()
#     return prompt

# def prepare_inputs_system_conversational(content, content_idx, use_demonstrations, demonstration_selection, demonstration_distribution, support_annots, sim_matrix, labels, k):
#     prompt_conv = []
#     if use_demonstrations:
#         if demonstration_selection == "tf-idf":
#             similar_entries = tfidf_sampler(sim_matrix[content_idx], labels, k, selection=demonstration_distribution)
#             similar_entry_indices = [entry[0] for entry in similar_entries]
#             samples = [support_annots[index] for index in similar_entry_indices]
#         else:
#             raise NotImplementedError(f"demonstration selection '{demonstration_selection}' not found")

#         if demonstration_distribution == "equal":
#             pass
#         else:
#             for index, s in enumerate(samples):
#                 answer = "Hateful" if s['label'] == "hateful" or s['label'] == 1 else "Not Hateful"
#                 qn = QUESTION_MULTI_TURN_TEMPLATE.format(content=s['content'])
#                 ans = ANSWER_MULTI_TURN_TEMPLATE.format(rationale=s['rationale'], answer=answer)
#                 prompt_conv.append({"role": "user", "content": qn})
#                 prompt_conv.append({"role": "assistant", "content": ans})

#     qn = QUESTION_MULTI_TURN_TEMPLATE.format(content=content['content'])
#     prompt_conv.append({"role": "user", "content": qn})
    
#     prompt = [
#         {"role": "system", "content": SYSTEM_PROMPT}
#     ] + prompt_conv 

#     print(json.dumps(prompt, indent=2))
#     exit()
#     return prompt