"""
Retrieval evaluation runner: runs a batch of queries against the vector store
and computes Recall@K and MRR metrics. Outputs versioned JSON reports to the
retrieval/<version>/evaluations/<eval_version>/ directory.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Benchmark")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the JSON evaluation dataset"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="unified_corpus",
        help="Name for the vector database collection.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of documents to retrieve per query"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluations",
        help="Base evaluation output directory",
    )
    parser.add_argument(
        "--use_reranker", action="store_true", help="Use cross-encoder reranker"
    )
    parser.add_argument(
        "--use_hybrid", action="store_true", help="Use Hybrid Search (BM25 + Dense)"
    )
    args = parser.parse_args()

    from src.retrieving.eval_runner import run_evaluation_pipeline

    run_evaluation_pipeline(
        dataset_path=args.dataset,
        collection_name=args.collection_name,
        top_k=args.top_k,
        output_dir_base=args.output_dir,
        use_reranker=args.use_reranker,
        use_hybrid=args.use_hybrid,
    )


if __name__ == "__main__":
    main()
