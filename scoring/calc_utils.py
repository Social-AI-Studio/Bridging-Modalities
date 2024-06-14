import numpy as np
from typing import List

def label_apk(
        expected_target: int, 
        document_targets: List[List[int]], 
        k=10,
        debug=False,
    ):
    if len(document_targets) > k:
        document_targets = document_targets[:k]

    if debug:
        print(expected_target)
        print(document_targets)

    relevant_count = 0.0
    precisions = []

    for i, p in enumerate(document_targets):
        if p == expected_target:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
        
            if debug:
                print(f"p@{i}: {precision_at_i}")

    if precisions:
        average_precision = sum(precisions) / len(precisions)
    else:
        average_precision = 0.0

    if debug:
        print(f"ap@{k}: {average_precision}")
    return average_precision

def label_mapk(expected_targets, document_targets, k, debug=False):
    return np.mean([label_apk(a,p,k,debug) for a,p in zip(expected_targets, document_targets)])

def categories_apk(
        expected_targets, 
        document_targets, 
        k=10,
        debug=False,
    ):

    if len(document_targets) > k:
        document_targets = document_targets[:k -1]

    if debug:
        print(expected_targets)
        print(document_targets)
        
    # Determine if each document is relevant
    relevance = [1 if any(target in doc for target in expected_targets) else 0 for doc in document_targets]

    # Calculate precision at each relevant position
    precisions = []
    relevant_count = 0
    for i, rel in enumerate(relevance):
        if rel == 1:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precisions.append(precision_at_i)
        
            if debug:
                print(f"p@{i}: {precision_at_i}")

    # Calculate the average precision
    if precisions:
        average_precision = sum(precisions) / len(precisions)
    else:
        average_precision = 0.0

    if debug:
        print(f"ap@{k}: {average_precision}")
    return average_precision

def categories_mapk(expected_targets, document_targets, k, debug=False):
    return np.mean([categories_apk(a,p,k,debug) for a,p in zip(expected_targets, document_targets)])


if __name__ == "__main__":
    expected_targets = [1]
    document_targets = [[1,0,0,1,1,0]]
    print(label_mapk(expected_targets, document_targets, k=6, debug=True))

    expected_targets = [[1, 2]]
    document_targets = [[[1], [0], [0], [1], [1, 2], [-1]]]
    print(categories_mapk(expected_targets, document_targets, k=6, debug=True))