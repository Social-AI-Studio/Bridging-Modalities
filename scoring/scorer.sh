LATENT_HATRED=/mnt/data1/datasets/hatespeech/latent_hatred/projects/CMTL-RAG/annotations/annotations.jsonl
MISOGYNISTIC_MEME=/mnt/data1/datasets/memes/Misogynistic_MEME/annotations/explanation.jsonl

FHM=/mnt/data1/datasets/memes/fhm_finegrained/annotations/dev_seen.json
FHM_CAPS=/mnt/data1/datasets/memes/fhm/captions/img_clean/ofa-large-caption/

MAMI=/mnt/data1/datasets/memes/mami/annotations/test.jsonl
MAMI_CAPS=/mnt/data1/datasets/memes/mami/captions/deepfillv2/test/ofa-large-caption/

for metric in textcaption2text textcaption2rationale
do
    echo "FHM - TFIDF - ${metric}"
    python3 retrieval_scorer.py \
        --annotation_filepath $FHM \
        --caption_dir $FHM_CAPS \
        --sim_matrix_filepath /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/fhm_lh_tfidf_matching.npy
        
    echo ""

    echo "FHM - BM25 - ${metric}"
    python3 retrieval_scorer.py \
        --annotation_filepath $FHM \
        --caption_dir $FHM_CAPS \
        --sim_matrix_filepath  /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/fhm_lh_bm25_matching.npy
    echo ""

    echo "MAMI - TFIDF - ${metric}"
    python3 retrieval_scorer.py \
        --annotation_filepath $MAMI \
        --caption_dir $MAMI_CAPS \
        --sim_matrix_filepath  /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/mami_lh_tfidf_matching.npy
    echo ""

    echo "MAMI - BM25 - ${metric}"
    python3 retrieval_scorer.py \
        --annotation_filepath $MAMI \
        --caption_dir $MAMI_CAPS \
        --sim_matrix_filepath  /mnt/data1/datasets/memes/cmtl-rag/sim_matrices_finalized/$metric/mami_lh_bm25_matching.npy
    echo ""
done