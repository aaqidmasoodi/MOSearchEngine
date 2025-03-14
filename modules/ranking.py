import math
from collections import defaultdict
import numpy as np
from .preprocess import preprocess_text 

# VectorSpaceModel Class - Implements a vector space model for ranking documents
class VectorSpaceModel:
    def __init__(self, documents, term_idf, doc_vectors):
        # Initialize with documents, term IDF values, and document vectors (TF-IDF)
        self.documents = documents
        self.term_idf = term_idf
        self.doc_vectors = doc_vectors

    # Internal method to compute the query vector based on the given query terms
    def _compute_query_vector(self, query_terms):
        query_vector = {}
        term_freq = defaultdict(int)
        for term in query_terms:
            term_freq[term] += 1  # Count term frequency in the query
        
        sum_squares = 0  # For normalization (cosine similarity)
        for term, tf in term_freq.items():
            if term in self.term_idf:
                log_tf = 1 + math.log(1 + tf)  # Use sublinear term frequency for queries
                query_vector[term] = log_tf * self.term_idf[term]  # TF-IDF weighting for query terms
                sum_squares += query_vector[term] ** 2  # Accumulate for normalization
        
        if sum_squares > 0:
            norm_factor = math.sqrt(sum_squares)  # Normalize the query vector
            for term in query_vector:
                query_vector[term] /= norm_factor  # Normalize each term's weight
        
        return query_vector

    # Ranks documents based on cosine similarity with the query
    def rank_documents(self, query):
        query_terms = preprocess_text(query)  # Preprocess query terms
        query_vector = self._compute_query_vector(query_terms)  # Compute query vector
        
        if not query_vector:
            return [(doc_id, 0) for doc_id in self.documents]  # Return default ranking if no query terms
        
        ranked_docs = []
        # For each document, compute similarity with the query vector
        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = sum(weight * doc_vector.get(term, 0) for term, weight in query_vector.items())
            ranked_docs.append((doc_id, similarity))  # Store document id and similarity score
        
        ranked_docs.sort(key=lambda x: x[1], reverse=True)  # Sort documents by similarity in descending order
        
        # Debug: Print top 10 documents for query
        print(f"\nVSM Top 10 for query '{query}':")
        for doc_id, score in ranked_docs[:10]:
            print(f"Doc {doc_id}: {score:.4f}")
        
        return ranked_docs


# BM25 Class - Implements the BM25 ranking function for documents
class BM25:
    def __init__(self, documents, k1=1.2, b=0.5):  # Tuned parameters for BM25
        self.documents = documents
        self.k1 = k1  # Term frequency scaling factor
        self.b = b    # Document length normalization factor
        
        from modules.preprocess import compute_doc_term_freqs  # Preprocess module to compute term frequencies
        self.doc_term_freqs, self.doc_lengths, self.avg_doc_length = compute_doc_term_freqs(documents)
        
        self.inverted_index = defaultdict(dict)
        # Build an inverted index for fast term lookup across documents
        for doc_id, term_freqs in self.doc_term_freqs.items():
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq
        
        self.total_docs = len(documents)  # Total number of documents
        self.term_idf = self._compute_idf()  # Compute IDF for terms

    # Computes Inverse Document Frequency (IDF) for each term
    def _compute_idf(self):
        term_idf = {}
        for term, doc_freq_dict in self.inverted_index.items():
            doc_freq = len(doc_freq_dict)  # How many documents contain the term
            # IDF formula with smoothing (adjusted for small document counts)
            term_idf[term] = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return term_idf

    # Compute BM25 score for a document based on the query terms
    def _compute_bm25_score(self, doc_id, query_terms):
        if doc_id not in self.doc_term_freqs:
            return 0  # No relevant terms, return score of 0
        
        score = 0
        doc_term_freq = self.doc_term_freqs[doc_id]  # Get document's term frequencies
        doc_length = self.doc_lengths[doc_id]  # Length of the document
        
        for term in set(query_terms):  # Avoid repeating terms in the query
            if term in self.term_idf:  # Check if term has a valid IDF
                tf = doc_term_freq.get(term, 0)  # Get term frequency for document
                idf = self.term_idf[term]  # Get IDF for the term
                
                # BM25 score formula components
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                # Compute individual term score and adjust by query term frequency
                term_score = idf * (numerator / denominator)
                term_weight = query_terms.count(term)
                score += term_score * term_weight  # Add to the total score for the document
        
        return score

    # Ranks documents using BM25 scoring
    def rank_documents(self, query):
        query_terms = preprocess_text(query)  # Preprocess query terms
        if not query_terms:
            return [(doc_id, 0) for doc_id in self.documents]  # Return default if no query terms
        
        relevant_docs = set()
        # Find all documents that have at least one term from the query
        for term in query_terms:
            if term in self.inverted_index:
                relevant_docs.update(self.inverted_index[term].keys())
        
        # Compute BM25 scores for relevant documents
        ranked_docs = [(doc_id, self._compute_bm25_score(doc_id, query_terms)) for doc_id in relevant_docs]
        scored_docs = {doc_id for doc_id, _ in ranked_docs}
        
        # Add documents that aren't in the relevant_docs set with a score of 0
        for doc_id in self.documents:
            if doc_id not in scored_docs:
                ranked_docs.append((doc_id, 0))
        
        ranked_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by BM25 score
        
        # Debug: Print top 10 documents for query
        print(f"\nBM25 Top 10 for query '{query}':")
        for doc_id, score in ranked_docs[:10]:
            print(f"Doc {doc_id}: {score:.4f}")
        
        return ranked_docs


# LanguageModel Class - Implements a language model for document ranking using smoothing
class LanguageModel:
    def __init__(self, documents, mu=500):  # 'mu' is the smoothing parameter
        self.documents = documents
        self.mu = mu  # Smoothing parameter to balance document-specific term probability
        
        from modules.preprocess import compute_doc_term_freqs  # Preprocess to compute term frequencies
        self.doc_term_freqs, self.doc_lengths, _ = compute_doc_term_freqs(documents)
        
        # Initialize collection statistics
        self.collection_term_freq = defaultdict(int)
        self.collection_length = 0
        
        # Accumulate term frequencies across the entire document collection
        for doc_id, term_freqs in self.doc_term_freqs.items():
            for term, freq in term_freqs.items():
                self.collection_term_freq[term] += freq
                self.collection_length += freq  # Total number of terms in the collection

    # Compute Language Model score for a document
    def _compute_lm_score(self, doc_id, query_terms):
        if doc_id not in self.doc_term_freqs:
            return float('-inf')  # Return negative infinity for unknown documents
        
        doc_term_freq = self.doc_term_freqs[doc_id]  # Get term frequencies for the document
        doc_length = self.doc_lengths[doc_id]  # Document length
        
        if doc_length == 0:
            return float('-inf')  # Avoid division by zero
        
        log_score = 0
        # For each query term, compute the smoothed probability for the document
        for term in query_terms:
            tf = doc_term_freq.get(term, 0)  # Term frequency in document
            p_term_collection = self.collection_term_freq.get(term, 0) / max(1, self.collection_length)  # Collection frequency of term
            smoothed_prob = (tf + self.mu * p_term_collection) / (doc_length + self.mu)  # Apply smoothing formula
            smoothed_prob = max(smoothed_prob, 1e-10)  # Avoid log of zero
            log_score += math.log(smoothed_prob)  # Add log probability to the score
        
        return log_score

    # Rank documents based on Language Model scoring
    def rank_documents(self, query):
        query_terms = preprocess_text(query)  # Preprocess query terms
        if not query_terms:
            return [(doc_id, 0) for doc_id in self.documents]  # Return default if no query terms
        
        ranked_docs = [(doc_id, self._compute_lm_score(doc_id, query_terms)) for doc_id in self.documents]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by score
        
        # Debug: Print top 10 documents for query
        print(f"\nLM Top 10 for query '{query}':")
        for doc_id, score in ranked_docs[:10]:
            print(f"Doc {doc_id}: {score:.4f}")
        
        return ranked_docs
