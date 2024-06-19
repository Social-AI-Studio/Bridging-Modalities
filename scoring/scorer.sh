LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM=/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json
FHM_CAPS=/mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/

MAMI=/mnt/data1/datasets/memes/mami/annotations/test.jsonl
MAMI_CAPS=/mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/

echo "FHM - RANDOM"
python3 retrieval_scorer.py \
    --annotation_filepath $FHM \
    --caption_dir $FHM_CAPS \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs ""  \
    --demonstration_selection "random" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/500_8500_matching.npy
    
echo ""

echo "MAMI - RANDOM"
python3 retrieval_scorer.py \
    --annotation_filepath $MAMI \
    --caption_dir $MAMI_CAPS \
    --support_filepaths $LATENT_HATRED \
    --support_caption_dirs "" \
    --support_feature_dirs ""  \
    --demonstration_selection "random" \
    --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/random/1000_8500_matching.npy
echo ""

# for metric in textcaption2text textcaption2rationale
# do
#     echo "FHM - TFIDF - ${metric}"
#     python3 retrieval_scorer.py \
#         --annotation_filepath $FHM \
#         --caption_dir $FHM_CAPS \
#         --support_filepaths $LATENT_HATRED \
#         --support_caption_dirs "" \
#         --support_feature_dirs ""  \
#         --demonstration_selection "tfidf" \
#         --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/fhm_lh_tfidf_matching.npy
        
#     echo ""

#     echo "FHM - BM25 - ${metric}"
#     python3 retrieval_scorer.py \
#         --annotation_filepath $FHM \
#         --caption_dir $FHM_CAPS \
#         --support_filepaths $LATENT_HATRED \
#         --support_caption_dirs "" \
#         --support_feature_dirs ""  \
#         --demonstration_selection "bm25" \
#         --sim_matrix_filepath  /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/fhm_lh_bm25_matching.npy
#     echo ""

#     echo "MAMI - TFIDF - ${metric}"
#     python3 retrieval_scorer.py \
#         --annotation_filepath $MAMI \
#         --caption_dir $MAMI_CAPS \
#         --support_filepaths $LATENT_HATRED \
#         --support_caption_dirs "" \
#         --support_feature_dirs ""  \
#         --demonstration_selection "tfidf" \
#         --sim_matrix_filepath  /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/mami_lh_tfidf_matching.npy
#     echo ""

#     echo "MAMI - BM25 - ${metric}"
#     python3 retrieval_scorer.py \
#         --annotation_filepath $MAMI \
#         --caption_dir $MAMI_CAPS \
#         --support_filepaths $LATENT_HATRED \
#         --support_caption_dirs "" \
#         --support_feature_dirs ""  \
#         --demonstration_selection "bm25" \
#         --sim_matrix_filepath  /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/mami_lh_bm25_matching.npy
#     echo ""
# done