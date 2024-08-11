
import argparse
import tqdm
import os
import json
import random
import numpy as np

def main(
    num_records: int,
    dataset_length: int,
    support_matrix_filepath: str,
):

    random_sim_matrix = []
    indices = list(range(dataset_length))
    print(len(indices))
    for i in range(num_records):
        vector = random.sample(indices, 32)
        random_sim_matrix.append(vector)

    random_sim_matrix = np.array(random_sim_matrix)
    print(random_sim_matrix.shape)
    with open(support_matrix_filepath, 'wb') as f:
        np.save(f, random_sim_matrix)
        np.save(f, np.array([]))
        np.save(f, np.array([]))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("CMTL RAG Baseline")
    # parser.add_argument("--dataset_length", type=int, default=29972)
    # parser.add_argument("--output_sim_matrix_filepath", type=str)

    # args = parser.parse_args()

    # main(
    #     args.
    #     args.dataset_length,
    #     args.sim_matrix_filepath,
    # )

    random.seed(2024)
    main(
        500,
        8500,
        "/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/500_8500_matching.npy"
    )

    random.seed(2024)
    main(
        1000,
        8500,
        "/mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/1000_8500_matching.npy"
    )
