# Bridging Modalities

This is the implementation of the paper **Bridging Modalities: Enhancing Cross-Modality Hate Speech Detection with Few-Shot In-Context Learning** (EMNLP 2024).

## Overview

Hate speech manifests in various forms, including text-based tweets and vision-language memes. While text-based hate speech is well-studied and has abundant resources, vision-language memes are less explored and lack available dataset resources.

Given that hate speech is a common underlying concept across modalities, this paper investigates the potential for **cross-modality knowledge transfer** to enhance the detection of hateful vision-language memes in a zero-shot setting.

## Data Pre-processing
To replicate the results, you need to pre-process data in the following steps:

1. Generate captions over each meme image.
2. Generates rationales for the support set memes using the content and ground truth label
3. Content similarity matrix generation 

**Caption Generation**: To perform hateful meme classification with the large language models, we perform image captioning on the meme using the [OFA model](https://github.com/OFA-Sys/OFA) pre-trained on the MSCOCO dataset.

**Rationale Generation**: We use `mistralai/Mistral-7B-Instruct-v0.3` to generate informative rationales that clarify the underlying meaning of the content, providing additional context for few-shot in-context learning. Specifically, the model generates rationales by taking the content and ground truth labels as input (`prompt + content → ground truth label → explanation`). Detailed information on the generation settings and few-shot templates for each support set can be found in the `rationale-generation` folder.

**Content Similarity Matrix Generation**: We use three sampling strategies: Random, TF-IDF, and BM-25. The TF-IDF and BM-25 approaches use text and caption information from test records to find similar examples in the support set. You can match test record `text` or `text+caption` with support set records’ `text+caption` or `rationale`. To avoid repeated calculations, scripts in the `matching` folder are available to generate and store these similarity values for different sampling strategies.

## Run Bridging-Modalities
Our project is built with 
- CUDA 12.2 (cuda_12.2.r12.2/compiler.33191640_0)
- Python 3.9.19

To set up the environment, run:
```
pip install -r requirements.txt
```
To replicate the reported performance for each model, navigate to the model's folder under the `baselines/scripts/inference` folder. Each setting (random, 0-shot, few-shot) is contained in a different script. To evaluate `Qwen/Qwen2-7B-Instruct` on using Latent Hatred as a support set using test record text + caption to support set record rationale matching run:
```bash
bash qwen-fs-lh-support.sh textcaption2rationale 
```
More information about the prompt template for each setting can be found in `baseslines/prompt_utils.py`

## Citation

Please cite our paper if you use Bridging-Modalities in your work:

```
@misc{hee2024bridgingmodalitiesenhancingcrossmodality,
      title={Bridging Modalities: Enhancing Cross-Modality Hate Speech Detection with Few-Shot In-Context Learning}, 
      author={Ming Shan Hee and Aditi Kumaresan and Roy Ka-Wei Lee},
      year={2024},
      eprint={2410.05600},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.05600}, 
}
```
