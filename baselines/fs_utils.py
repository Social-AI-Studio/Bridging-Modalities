# from matching.tfidf_wrapper import get_top_k_similar as tfidf_sampler
# from matching.bm25_wrapper import get_top_k_similar as bm25_sampler

from tfidf_wrapper import get_top_k_similar as tfidf_sampler
from bm25_wrapper import get_top_k_similar as bm25_sampler

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
        prepare_fn = prepare_inputs_single_prompt
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

    return prompt
