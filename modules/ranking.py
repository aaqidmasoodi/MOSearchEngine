# ranking.py
import math
from collections import defaultdict
import numpy as np
from preprocess import preprocess_text

class VectorSpaceModel:
    """Implements the Vector Space Model (VSM) for ranking documents."""
    
    def __init__(self, documents, term_idf, doc_vectors):
        self.documents = documents
        self.term_idf = term_idf
        self.doc_vectors = doc_vectors

    def _compute_query_vector(self, query_terms):
        """Computes the TF-IDF vector for the query with improved weighting."""
        query_vector = {}
        term_freq = defaultdict(int)
        for term in query_terms:
            term_freq[term] += 1
        
        # Create and normalize query vector
        sum_squares = 0
        
        for term, tf in term_freq.items():
            if term in self.term_idf:
                # Apply log normalization to term frequency
                log_tf = 1 + math.log(tf) if tf > 0 else 0
                query_vector[term] = log_tf * self.term_idf[term]
                sum_squares += query_vector[term] ** 2
        
        # L2 normalize the query vector
        if sum_squares > 0:
            norm_factor = math.sqrt(sum_squares)
            for term in query_vector:
                query_vector[term] /= norm_factor
                
        return query_vector

    def rank_documents(self, query):
        """Ranks documents based on cosine similarity with the query."""
        query_terms = preprocess_text(query)
        query_vector = self._compute_query_vector(query_terms)
        
        if not query_vector:  # Handle empty queries
            return [(doc_id, 0) for doc_id in self.documents]
        
        ranked_docs = []
        for doc_id, doc_vector in self.doc_vectors.items():
            # Compute dot product directly for cosine similarity
            # (vectors are already normalized)
            similarity = 0
            for term, weight in query_vector.items():
                if term in doc_vector:
                    similarity += weight * doc_vector[term]
            
            ranked_docs.append((doc_id, similarity))
        
        # Sort documents by similarity score in descending order
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs


class BM25:
    """Implements the BM25 ranking algorithm with improved efficiency."""
    
    def __init__(self, documents, k1=1.2, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        # Precompute everything we need for efficiency
        from preprocess import compute_doc_term_freqs
        self.doc_term_freqs, self.doc_lengths, self.avg_doc_length = compute_doc_term_freqs(documents)
        
        # Build inverted index with term frequencies
        self.inverted_index = defaultdict(dict)
        for doc_id, term_freqs in self.doc_term_freqs.items():
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq
        
        self.total_docs = len(documents)
        self.term_idf = self._compute_idf()

    def _compute_idf(self):
        """Computes IDF (Inverse Document Frequency) for each term with improved formula."""
        term_idf = {}
        for term, doc_freq_dict in self.inverted_index.items():
            doc_freq = len(doc_freq_dict)
            # BM25 IDF formula with smoothing to handle edge cases
            term_idf[term] = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return term_idf

    def _compute_bm25_score(self, doc_id, query_terms):
        """Computes the BM25 score for a document and query with improved efficiency."""
        if doc_id not in self.doc_term_freqs:
            return 0
            
        score = 0
        doc_term_freq = self.doc_term_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        for term in set(query_terms):  # Use set to avoid counting query terms multiple times
            if term in self.term_idf:
                tf = doc_term_freq.get(term, 0)
                idf = self.term_idf[term]
                
                # Standard BM25 term scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                term_score = idf * (numerator / denominator)
                
                # Weight query terms by their frequency in the query
                term_weight = query_terms.count(term)
                score += term_score * term_weight
                
        return score

    def rank_documents(self, query):
        """Ranks documents based on BM25 scores with query term weighting."""
        query_terms = preprocess_text(query)
        
        if not query_terms:  # Handle empty queries
            return [(doc_id, 0) for doc_id in self.documents]
            
        # Find relevant documents first (those containing at least one query term)
        relevant_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                relevant_docs.update(self.inverted_index[term].keys())
        
        # Only score relevant documents for efficiency
        ranked_docs = []
        for doc_id in relevant_docs:
            score = self._compute_bm25_score(doc_id, query_terms)
            ranked_docs.append((doc_id, score))
        
        # Add any remaining documents with zero scores for completeness
        scored_docs = {doc_id for doc_id, _ in ranked_docs}
        for doc_id in self.documents:
            if doc_id not in scored_docs:
                ranked_docs.append((doc_id, 0))
        
        # Sort documents by BM25 score in descending order
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs


class LanguageModel:
    """Implements a unigram Language Model with Dirichlet smoothing for better performance."""
    
    def __init__(self, documents, mu=2000):
        self.documents = documents
        self.mu = mu  # Dirichlet smoothing parameter
        
        # Precompute term frequencies and document lengths
        from preprocess import compute_doc_term_freqs
        self.doc_term_freqs, self.doc_lengths, _ = compute_doc_term_freqs(documents)
        
        # Compute collection statistics
        self.collection_term_freq = defaultdict(int)
        self.collection_length = 0
        
        for doc_id, term_freqs in self.doc_term_freqs.items():
            for term, freq in term_freqs.items():
                self.collection_term_freq[term] += freq
                self.collection_length += freq

    def _compute_lm_score(self, doc_id, query_terms):
        """Computes the Language Model score using Dirichlet smoothing and log probabilities."""
        if doc_id not in self.doc_term_freqs:
            return float('-inf')  # Document doesn't exist
            
        doc_term_freq = self.doc_term_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        if doc_length == 0:
            return float('-inf')  # Empty document

        # Use log probabilities to avoid underflow
        log_score = 0
        
        for term in query_terms:
            # Term frequency in document
            tf = doc_term_freq.get(term, 0)
            
            # Collection probability (smoothing background)
            p_term_collection = self.collection_term_freq.get(term, 0) / max(1, self.collection_length)
            
            # Dirichlet smoothed probability
            smoothed_prob = (tf + self.mu * p_term_collection) / (doc_length + self.mu)
            
            # Add small epsilon to prevent log(0)
            smoothed_prob = max(smoothed_prob, 1e-10)
            
            # Add log probability to score
            log_score += math.log(smoothed_prob)
        
        return log_score

    def rank_documents(self, query):
        """Ranks documents based on Language Model log probabilities."""
        query_terms = preprocess_text(query)
        
        if not query_terms:  # Handle empty queries
            return [(doc_id, 0) for doc_id in self.documents]
        
        ranked_docs = []
        for doc_id in self.documents:
            score = self._compute_lm_score(doc_id, query_terms)
            ranked_docs.append((doc_id, score))
        
        # Sort documents by Language Model score in descending order
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return ranked_docs