import os
import time
import argparse
from multiprocessing import Pool, cpu_count
from modules.parser import parse_documents, parse_queries, parse_qrels  # Import functions for parsing datasets
from modules.ranking import VectorSpaceModel, BM25, LanguageModel  # Import ranking models
from modules.indexing import build_inverted_index, save_index, load_index  # Import functions for indexing documents

# Function to save the ranked documents in the TREC evaluation format
def save_trec_eval_output(ranked_docs, query_id, model_name, output_file):
    with open(output_file, "a") as f:
        # Write the top 1000 ranked documents to the output file
        for rank, (doc_id, score) in enumerate(ranked_docs[:1000], start=1):
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {model_name}\n")

# Function to evaluate a single query using multiple ranking models (VSM, BM25, LM)
def evaluate_single_query(args):
    query_id, query_text, vsm, bm25, lm, output_dir = args
    try:
        start_time = time.time()
        # Rank documents using VSM model
        ranked_docs_vsm = vsm.rank_documents(query_text)
        vsm_time = time.time() - start_time

        start_time = time.time()
        # Rank documents using BM25 model
        ranked_docs_bm25 = bm25.rank_documents(query_text)
        bm25_time = time.time() - start_time

        start_time = time.time()
        # Rank documents using Language Model (LM)
        ranked_docs_lm = lm.rank_documents(query_text)
        lm_time = time.time() - start_time

        # Save results for each model in TREC format
        save_trec_eval_output(ranked_docs_vsm, query_id, "vsm", os.path.join(output_dir, "vsm_results.txt"))
        save_trec_eval_output(ranked_docs_bm25, query_id, "bm25", os.path.join(output_dir, "bm25_results.txt"))
        save_trec_eval_output(ranked_docs_lm, query_id, "lm", os.path.join(output_dir, "lm_results.txt"))

        # Print time taken for each model
        print(f"Processed query {query_id} | VSM: {vsm_time:.2f}s, BM25: {bm25_time:.2f}s, LM: {lm_time:.2f}s")
        return True
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        return False

# Function to evaluate all queries with the three ranking models (VSM, BM25, LM)
def evaluate_models(documents, queries, qrels, output_dir="results", use_cached_index=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the output directory if it doesn't exist

    index_file = "index_data.pkl"
    index_data = load_index(index_file) if use_cached_index else None  # Load cached index if available
    
    if not index_data:
        print("Preprocessing documents...")
        start_time = time.time()
        # Build inverted index and calculate necessary values
        index_data = build_inverted_index(documents)
        print(f"Preprocessing completed in {time.time() - start_time:.2f}s")
        save_index(index_data, index_file)  # Save index for future use
    
    inverted_index = index_data['inverted_index']
    term_idf = index_data['idf_values']
    doc_vectors = index_data['doc_vectors']
    
    print("Initializing ranking models...")
    start_time = time.time()
    # Initialize ranking models (VSM, BM25, LM)
    vsm = VectorSpaceModel(documents, term_idf, doc_vectors)
    bm25 = BM25(documents, k1=1.2, b=0.5)  # Parameters for BM25
    lm = LanguageModel(documents, mu=500)  # Parameter for Language Model
    print(f"Ranking models initialized in {time.time() - start_time:.2f}s")

    # Initialize result files by clearing or creating them
    for file in ["vsm_results.txt", "bm25_results.txt", "lm_results.txt"]:
        open(os.path.join(output_dir, file), "w").close()

    # Prepare arguments for multiprocessing (query_id, query_text, models, and output directory)
    args = [(query_id, query_text, vsm, bm25, lm, output_dir) for query_id, query_text in queries.items()]
    num_processes = min(cpu_count(), len(queries))  # Limit number of processes to available CPUs or number of queries
    
    print(f"Evaluating {len(queries)} queries using {num_processes} CPU cores...")
    start_time = time.time()
    
    # Only process the first 5 queries for debugging purposes, else process all
    success_count = 0
    with Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_single_query, args[:5] if len(queries) > 5 else args)
        success_count = sum(1 for r in results if r)  # Count successfully processed queries
    
    print(f"Processed {success_count}/{len(queries)} queries in {time.time() - start_time:.2f}s")
    print(f"Results saved in '{output_dir}'")

# Function to run TREC evaluation on the results
def run_trec_eval(qrels_file, results_dir="results", metrics=None):
    if metrics is None:
        metrics = ["map", "P.5", "ndcg"]  # Default metrics for evaluation
    metrics_str = " ".join([f"-m {m}" for m in metrics])  # Format metrics for the command
    
    print("\n===== EVALUATION RESULTS =====")
    # Evaluate for each model (VSM, BM25, LM)
    for model, results_file in [("VSM", "vsm_results.txt"), ("BM25", "bm25_results.txt"), ("LM", "lm_results.txt")]:
        results_path = os.path.join(results_dir, results_file)
        if os.path.exists(results_path):  # Check if the result file exists
            print(f"\nEvaluating {model}...")
            # Run the TREC evaluation command
            cmd = f"trec_eval {metrics_str} {qrels_file} {results_path}"
            exit_code = os.system(cmd)  # Execute the command in the shell
            if exit_code != 0:
                print(f"Warning: trec_eval failed with exit code {exit_code}")
        else:
            print(f"Results file for {model} not found: {results_path}")

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Information Retrieval System Evaluation")
    parser.add_argument("--docs", default="datasets/cran.all.1400.xml")  # Input document file
    parser.add_argument("--queries", default="datasets/cran.qry.xml")  # Input query file
    parser.add_argument("--qrels", default="datasets/cranqrel.trec.txt")  # Input relevance file
    parser.add_argument("--output", default="results")  # Directory for output results
    parser.add_argument("--no-cache", action="store_true")  # Option to disable cache
    parser.add_argument("--metrics", nargs="+", default=["map", "P.5", "ndcg"])  # Evaluation metrics
    return parser.parse_args()

# Main function that drives the evaluation and runs TREC evaluation
if __name__ == "__main__":
    try:
        args = parse_args()  # Parse command-line arguments
        documents = parse_documents(args.docs)  # Parse the documents
        queries = parse_queries(args.queries)  # Parse the queries
        qrels = parse_qrels(args.qrels)  # Parse the relevance judgments
        evaluate_models(documents, queries, qrels, args.output, not args.no_cache)  # Evaluate models on queries
        run_trec_eval(args.qrels, args.output, args.metrics)  # Run TREC evaluation on the results
    except Exception as e:
        print(f"Fatal error: {e}")
