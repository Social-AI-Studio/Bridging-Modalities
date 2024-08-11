# from matching.tfidf_wrapper import get_top_k_similar as tfidf_sampler
# from matching.bm25_wrapper import get_top_k_similar as bm25_sampler

from tfidf_wrapper import get_top_k_similar as tfidf_sampler
from bm25_wrapper import get_top_k_similar as bm25_sampler

from prompt_utils import (
    INTRODUCTION,
    EXAMPLE_TEMPLATE_WITH_RATIONALE,
    QUESTION_TEMPLATE,
)

from llava.constants import IMAGE_PLACEHOLDER

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
