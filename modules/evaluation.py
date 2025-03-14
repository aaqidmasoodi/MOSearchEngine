import os
import time
import argparse
from multiprocessing import Pool, cpu_count
from parser import parse_documents, parse_queries, parse_qrels
from ranking import VectorSpaceModel, BM25, LanguageModel
from preprocess import preprocess_all
from indexing import build_inverted_index, save_index, load_index

def save_trec_eval_output(ranked_docs, query_id, model_name, output_file):
    with open(output_file, "a") as f:
        for rank, (doc_id, score) in enumerate(ranked_docs[:1000], start=1):
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {model_name}\n")

def evaluate_single_query(args):
    query_id, query_text, vsm, bm25, lm, output_dir = args
    try:
        start_time = time.time()
        ranked_docs_vsm = vsm.rank_documents(query_text)
        vsm_time = time.time() - start_time

        start_time = time.time()
        ranked_docs_bm25 = bm25.rank_documents(query_text)
        bm25_time = time.time() - start_time

        start_time = time.time()
        ranked_docs_lm = lm.rank_documents(query_text)
        lm_time = time.time() - start_time

        save_trec_eval_output(ranked_docs_vsm, query_id, "vsm", os.path.join(output_dir, "vsm_results.txt"))
        save_trec_eval_output(ranked_docs_bm25, query_id, "bm25", os.path.join(output_dir, "bm25_results.txt"))
        save_trec_eval_output(ranked_docs_lm, query_id, "lm", os.path.join(output_dir, "lm_results.txt"))

        print(f"Processed query {query_id} | VSM: {vsm_time:.2f}s, BM25: {bm25_time:.2f}s, LM: {lm_time:.2f}s")
        return True
    except Exception as e:
        print(f"Error processing query {query_id}: {e}")
        return False

def evaluate_models(documents, queries, qrels, output_dir="results", use_cached_index=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index_file = "index_data.pkl"
    index_data = load_index(index_file) if use_cached_index else None
    
    if not index_data:
        print("Preprocessing documents...")
        start_time = time.time()
        index_data = build_inverted_index(documents)
        print(f"Preprocessing completed in {time.time() - start_time:.2f}s")
        save_index(index_data, index_file)
    
    inverted_index = index_data['inverted_index']
    term_idf = index_data['idf_values']
    doc_vectors = index_data['doc_vectors']
    
    print("Initializing ranking models...")
    start_time = time.time()
    vsm = VectorSpaceModel(documents, term_idf, doc_vectors)
    bm25 = BM25(documents, k1=1.2, b=0.5)  # Tuned parameters
    lm = LanguageModel(documents, mu=500)  # Tuned parameter
    print(f"Ranking models initialized in {time.time() - start_time:.2f}s")

    for file in ["vsm_results.txt", "bm25_results.txt", "lm_results.txt"]:
        open(os.path.join(output_dir, file), "w").close()

    args = [(query_id, query_text, vsm, bm25, lm, output_dir) for query_id, query_text in queries.items()]
    num_processes = min(cpu_count(), len(queries))
    
    print(f"Evaluating {len(queries)} queries using {num_processes} CPU cores...")
    start_time = time.time()
    
    # Limit debug output to first 5 queries
    success_count = 0
    with Pool(processes=num_processes) as pool:
        results = pool.map(evaluate_single_query, args[:5] if len(queries) > 5 else args)
        success_count = sum(1 for r in results if r)
    
    print(f"Processed {success_count}/{len(queries)} queries in {time.time() - start_time:.2f}s")
    print(f"Results saved in '{output_dir}'")

def run_trec_eval(qrels_file, results_dir="results", metrics=None):
    if metrics is None:
        metrics = ["map", "P.5", "ndcg"]
    metrics_str = " ".join([f"-m {m}" for m in metrics])
    
    print("\n===== EVALUATION RESULTS =====")
    for model, results_file in [("VSM", "vsm_results.txt"), ("BM25", "bm25_results.txt"), ("LM", "lm_results.txt")]:
        results_path = os.path.join(results_dir, results_file)
        if os.path.exists(results_path):
            print(f"\nEvaluating {model}...")
            cmd = f"trec_eval {metrics_str} {qrels_file} {results_path}"
            exit_code = os.system(cmd)
            if exit_code != 0:
                print(f"Warning: trec_eval failed with exit code {exit_code}")
        else:
            print(f"Results file for {model} not found: {results_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Information Retrieval System Evaluation")
    parser.add_argument("--docs", default="datasets/cran.all.1400.xml")
    parser.add_argument("--queries", default="datasets/cran.qry.xml")
    parser.add_argument("--qrels", default="datasets/cranqrel.trec.txt")
    parser.add_argument("--output", default="results")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=["map", "P.5", "ndcg"])
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        documents = parse_documents(args.docs)
        queries = parse_queries(args.queries)
        qrels = parse_qrels(args.qrels)
        evaluate_models(documents, queries, qrels, args.output, not args.no_cache)
        run_trec_eval(args.qrels, args.output, args.metrics)
    except Exception as e:
        print(f"Fatal error: {e}")