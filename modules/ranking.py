import math
from collections import defaultdict
import numpy as np
from preprocess import preprocess_text

class VectorSpaceModel:
    def __init__(self, documents, term_idf, doc_vectors):
        self.documents = documents
        self.term_idf = term_idf
        self.doc_vectors = doc_vectors

    def _compute_query_vector(self, query_terms):
        query_vector = {}
        term_freq = defaultdict(int)
        for term in query_terms:
            term_freq[term] += 1
        
        sum_squares = 0
        for term, tf in term_freq.items():
            if term in self.term_idf:
                log_tf = 1 + math.log(1 + tf)  # Sublinear TF for query
                query_vector[term] = log_tf * self.term_idf[term]
                sum_squares += query_vector[term] ** 2
        
        if sum_squares > 0:
            norm_factor = math.sqrt(sum_squares)
            for term in query_vector:
                query_vector[term] /= norm_factor
        return query_vector

    def rank_documents(self, query):
        query_terms = preprocess_text(query)
        query_vector = self._compute_query_vector(query_terms)
        
        if not query_vector:
            return [(doc_id, 0) for doc_id in self.documents]
        
        ranked_docs = []
        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = sum(weight * doc_vector.get(term, 0) for term, weight in query_vector.items())
            ranked_docs.append((doc_id, similarity))
        
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Print top 10
        print(f"\nVSM Top 10 for query '{query}':")
        for doc_id, score in ranked_docs[:10]:
            print(f"Doc {doc_id}: {score:.4f}")
        
        return ranked_docs

class BM25:
    def __init__(self, documents, k1=1.2, b=0.5):  # Tuned parameters
        self.documents = documents
        self.k1 = k1
        self.b = b
        from preprocess import compute_doc_term_freqs
        self.doc_term_freqs, self.doc_lengths, self.avg_doc_length = compute_doc_term_freqs(documents)
        
        self.inverted_index = defaultdict(dict)
        for doc_id, term_freqs in self.doc_term_freqs.items():
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq
        
        self.total_docs = len(documents)
        self.term_idf = self._compute_idf()

    def _compute_idf(self):
        term_idf = {}
        for term, doc_freq_dict in self.inverted_index.items():
            doc_freq = len(doc_freq_dict)
            term_idf[term] = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return term_idf

    def _compute_bm25_score(self, doc_id, query_terms):
        if doc_id not in self.doc_term_freqs:
            return 0
        score = 0
        doc_term_freq = self.doc_term_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        for term in set(query_terms):
            if term in self.term_idf:
                tf = doc_term_freq.get(term, 0)
                idf = self.term_idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                term_score = idf * (numerator / denominator)
                term_weight = query_terms.count(term)
                score += term_score * term_weight
        return score

    def rank_documents(self, query):
        query_terms = preprocess_text(query)
        if not query_terms:
            return [(doc_id, 0) for doc_id in self.documents]
        
        relevant_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                relevant_docs.update(self.inverted_index[term].keys())
        
        ranked_docs = [(doc_id, self._compute_bm25_score(doc_id, query_terms)) for doc_id in relevant_docs]
        scored_docs = {doc_id for doc_id, _ in ranked_docs}
        for doc_id in self.documents:
            if doc_id not in scored_docs:
                ranked_docs.append((doc_id, 0))
        
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Print top 10
        print(f"\nBM25 Top 10 for query '{query}':")
        for doc_id, score in ranked_docs[:10]:
            print(f"Doc {doc_id}: {score:.4f}")
        
        return ranked_docs

class LanguageModel:
    def __init__(self, documents, mu=500):  # Lower mu
        self.documents = documents
        self.mu = mu
        from preprocess import compute_doc_term_freqs
        self.doc_term_freqs, self.doc_lengths, _ = compute_doc_term_freqs(documents)
        
        self.collection_term_freq = defaultdict(int)
        self.collection_length = 0
        for doc_id, term_freqs in self.doc_term_freqs.items():
            for term, freq in term_freqs.items():
                self.collection_term_freq[term] += freq
                self.collection_length += freq

    def _compute_lm_score(self, doc_id, query_terms):
        if doc_id not in self.doc_term_freqs:
            return float('-inf')
        doc_term_freq = self.doc_term_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        if doc_length == 0:
            return float('-inf')
        
        log_score = 0
        for term in query_terms:
            tf = doc_term_freq.get(term, 0)
            p_term_collection = self.collection_term_freq.get(term, 0) / max(1, self.collection_length)
            smoothed_prob = (tf + self.mu * p_term_collection) / (doc_length + self.mu)
            smoothed_prob = max(smoothed_prob, 1e-10)
            log_score += math.log(smoothed_prob)
        return log_score

    def rank_documents(self, query):
        query_terms = preprocess_text(query)
        if not query_terms:
            return [(doc_id, 0) for doc_id in self.documents]
        
        ranked_docs = [(doc_id, self._compute_lm_score(doc_id, query_terms)) for doc_id in self.documents]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Print top 10
        print(f"\nLM Top 10 for query '{query}':")
        for doc_id, score in ranked_docs[:10]:
            print(f"Doc {doc_id}: {score:.4f}")
        
        return ranked_docs