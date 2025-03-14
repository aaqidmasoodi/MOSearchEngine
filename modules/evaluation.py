# evaluation.py
import os
import time
import argparse
from multiprocessing import Pool, cpu_count
from parser import parse_documents, parse_queries, parse_qrels
from ranking import VectorSpaceModel, BM25, LanguageModel
from preprocess import preprocess_all
from indexing import build_inverted_index, save_index, load_index

def save_trec_eval_output(ranked_docs, query_id, model_name, output_file):
    """Saves the ranked documents in the format required by trec_eval."""
    with open(output_file, "a") as f:
        for rank, (doc_id, score) in enumerate(ranked_docs[:1000], start=1):  # Limit to top 1000 documents per TREC standard
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {model_name}\n")

def evaluate_single_query(args):
    """Evaluates a single query using all three models."""
    query_id, query_text, vsm, bm25, lm, output_dir = args

    try:
        # Rank documents for the query using each model
        start_time = time.time()
        ranked_docs_vsm = vsm.rank_documents(query_text)
        vsm_time = time.time() - start_time

        start_time = time.time()
        ranked_docs_bm25 = bm25.rank_documents(query_text)
        bm25_time = time.time() - start_time

        start_time = time.time()
        ranked_docs_lm = lm.rank_documents(query_text)
        lm_time = time.time() - start_time

        # Save the results
        save_trec_eval_output(ranked_docs_vsm, query_id, "vsm", os.path.join(output_dir, "vsm_results.txt"))
        save_trec_eval_output(ranked_docs_bm25, query_id, "bm25", os.path.join(output_dir, "bm25_results.txt"))
        save_trec_eval_output(ranked_docs_lm, query_id, "lm", os.path.join(output_dir, "lm_results.txt"))

        # Log progress
        print(f"Processed query {query_id} | VSM: {vsm_time:.2f}s, BM25: {bm25_time:.2f}s, LM: {lm_time:.2f}s")
        return True
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        return False

def evaluate_models(documents, queries, qrels, output_dir="results", use_cached_index=True):
    """Evaluates the three ranking models (VSM, BM25, LM) and generates output files for trec_eval."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load cached index or build new one
    index_file = "index_data.pkl"
    index_data = None
    
    if use_cached_index:
        index_data = load_index(index_file)
    
    if not index_data:
        # Precompute values
        print("Preprocessing documents...")
        start_time = time.time()
        
        # Build comprehensive index data
        index_data = build_inverted_index(documents)
        print(f"Preprocessing completed in {time.time() - start_time:.2f}s")
        
        # Save index for future use
        save_index(index_data, index_file)
    
    # Extract needed components from index data
    inverted_index = index_data['inverted_index']
    term_idf = index_data['idf_values']
    doc_vectors = index_data['doc_vectors']
    
    # Initialize the ranking models
    print("Initializing ranking models...")
    start_time = time.time()
    vsm = VectorSpaceModel(documents, term_idf, doc_vectors)
    
    # Use k1=1.5, b=0.75 for BM25 (slightly modified parameters for better performance)
    bm25 = BM25(documents, k1=1.5, b=0.75)
    
    # Use mu=2000 for Language Model with Dirichlet smoothing
    lm = LanguageModel(documents, mu=2000)
    print(f"Ranking models initialized in {time.time() - start_time:.2f}s")

    # Clear the output files if they already exist
    for file in ["vsm_results.txt", "bm25_results.txt", "lm_results.txt"]:
        open(os.path.join(output_dir, file), "w").close()

    # Prepare arguments for multiprocessing
    args = [(query_id, query_text, vsm, bm25, lm, output_dir) for query_id, query_text in queries.items()]

    # Determine optimal number of processes (use at most available CPUs)
    num_processes = min(cpu_count(), len(queries))
    
    # Use multiprocessing to evaluate queries in parallel
    print(f"Evaluating {len(queries)} queries using {num_processes} CPU cores...")
    start_time = time.time()
    
    success_count = 0
    with Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_single_query, args)
        success_count = sum(1 for r in results if r)
    
    print(f"Successfully processed {success_count}/{len(queries)} queries in {time.time() - start_time:.2f}s")
    print(f"Results saved in the '{output_dir}' directory.")

def run_trec_eval(qrels_file, results_dir="results", metrics=None):
    """Runs the trec_eval tool to evaluate the results with specified metrics."""
    # Default metrics if none provided
    if metrics is None:
        metrics = ["map", "P.5", "ndcg", "recip_rank", "bpref"]
    
    # Build metrics string
    metrics_str = " ".join([f"-m {m}" for m in metrics])
    
    # Paths to the output files
    vsm_results_file = os.path.join(results_dir, "vsm_results.txt")
    bm25_results_file = os.path.join(results_dir, "bm25_results.txt")
    lm_results_file = os.path.join(results_dir, "lm_results.txt")

    print("\n===== EVALUATION RESULTS =====")
    
    # Run trec_eval for each model
    for model, results_file in [("VSM", vsm_results_file), ("BM25", bm25_results_file), ("LM", lm_results_file)]:
        if os.path.exists(results_file):
            print(f"\nEvaluating {model}...")
            cmd = f"trec_eval {metrics_str} {qrels_file} {results_file}"
            exit_code = os.system(cmd)
            
            if exit_code != 0:
                print(f"Warning: trec_eval command failed with exit code {exit_code}")
                print(f"Command was: {cmd}")
        else:
            print(f"Results file for {model} not found: {results_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Information Retrieval System Evaluation")
    
    parser.add_argument("--docs", default="datasets/cran.all.1400.xml",
                        help="Path to documents XML file")
    parser.add_argument("--queries", default="datasets/cran.qry.xml",
                        help="Path to queries XML file")
    parser.add_argument("--qrels", default="datasets/cranqrel.trec.txt",
                        help="Path to relevance judgments file")
    parser.add_argument("--output", default="results",
                        help="Output directory for results")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't use cached index")
    parser.add_argument("--metrics", nargs="+", 
                        default=["map", "P.5", "ndcg"],
                        help="Metrics to use in trec_eval")
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_args()
        
        # File paths
        cran_docs_file = args.docs
        cran_queries_file = args.queries
        cran_qrels_file = args.qrels
        output_dir = args.output
        use_cached_index = not args.no_cache
        evaluation_metrics = args.metrics

        # Parse the documents, queries, and relevance judgments
        print("Parsing documents...")
        documents = parse_documents(cran_docs_file)
        print("Parsing queries...")
        queries = parse_queries(cran_queries_file)
        print("Parsing relevance judgments...")
        qrels = parse_qrels(cran_qrels_file)

        # Evaluate the models and generate output files for trec_eval
        evaluate_models(documents, queries, qrels, output_dir, use_cached_index)

        # Run trec_eval to evaluate the results
        run_trec_eval(cran_qrels_file, output_dir, evaluation_metrics)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Partial results may have been saved.")
    except Exception as e:
        print(f"Fatal error during evaluation: {e}")