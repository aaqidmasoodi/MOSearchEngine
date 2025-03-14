import os
import argparse
from modules.evaluation import evaluate_models, run_trec_eval
from modules.parser import parse_documents, parse_queries, parse_qrels

def main():
    # Define default paths
    docs_path = os.path.join("datasets", "cran.all.1400.xml")
    queries_path = os.path.join("datasets", "cran.qry.xml")
    qrels_path = os.path.join("datasets", "cranqrel.trec.txt")
    output_dir = "results"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the MOSearchEngine evaluation.")
    parser.add_argument("--docs", default=docs_path, help="Path to the documents XML file.")
    parser.add_argument("--queries", default=queries_path, help="Path to the queries XML file.")
    parser.add_argument("--qrels", default=qrels_path, help="Path to the relevance judgments file.")
    parser.add_argument("--output", default=output_dir, help="Directory to save evaluation results.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of the inverted index.")
    parser.add_argument("--metrics", nargs="+", default=["map", "P.5", "ndcg"], help="Metrics to evaluate.")
    
    args = parser.parse_args()

    # Parse documents, queries, and relevance judgments
    documents = parse_documents(args.docs)
    queries = parse_queries(args.queries)
    qrels = parse_qrels(args.qrels)

    # Evaluate models
    evaluate_models(documents, queries, qrels, args.output, not args.no_cache)

    # Run TREC evaluation
    run_trec_eval(args.qrels, args.output, args.metrics)

if __name__ == "__main__":
    main()