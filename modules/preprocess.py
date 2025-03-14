# preprocess.py
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ssl
import nltk
import math

# Fix SSL certificate issue for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Download required NLTK data with error handling
def download_nltk_resources():
    """Download required NLTK resources with error handling."""
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {e}")

download_nltk_resources()

def preprocess_text(text):
    """Tokenizes, removes stopwords, and stems words with improved approach."""
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Convert to lowercase and tokenize
    words = word_tokenize(text.lower())
    
    # Filter words more carefully - keep some important short words
    words = [w for w in words if (w.isalnum() and w not in stop_words) or 
             (len(w) > 1 and w.isalpha())]
    
    # Apply stemming (consider keeping some common technical terms unstemmed)
    words = [stemmer.stem(w) if len(w) > 3 else w for w in words]
    
    return words

def build_vocabulary(documents):
    """Builds a vocabulary from all documents."""
    vocabulary = set()
    for doc_id, doc in documents.items():
        # Combine title and text
        title_words = preprocess_text(doc["title"])
        text_words = preprocess_text(doc["text"])
        
        # Add all words to vocabulary
        vocabulary.update(title_words)
        vocabulary.update(text_words)
    
    return vocabulary

def compute_doc_term_freqs(documents):
    """Precomputes term frequencies for all documents."""
    doc_term_freqs = {}
    doc_lengths = {}
    
    for doc_id, doc in documents.items():
        # Combine title and text with higher weight for title
        title_terms = preprocess_text(doc["title"])
        text_terms = preprocess_text(doc["text"])
        
        # Process all terms
        all_terms = title_terms * 2 + text_terms  # Give title terms more weight
        doc_lengths[doc_id] = len(all_terms)
        
        # Count term frequencies
        term_freq = defaultdict(int)
        for term in all_terms:
            term_freq[term] += 1
            
        doc_term_freqs[doc_id] = term_freq
    
    # Calculate average document length
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
    
    return doc_term_freqs, doc_lengths, avg_doc_length

# Note: This function is kept for backwards compatibility but should be considered deprecated
# Use build_inverted_index from indexing.py instead
def preprocess_all(documents):
    """
    Preprocesses all documents and returns precomputed values.
    
    DEPRECATED: Use build_inverted_index from indexing.py instead.
    This function is kept for backwards compatibility.
    """
    from indexing import build_inverted_index
    
    # Issue a deprecation warning
    import warnings
    warnings.warn(
        "preprocess_all() is deprecated. Use build_inverted_index() from indexing.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Call the more comprehensive indexing function
    index_data = build_inverted_index(documents)
    
    # Extract components needed for backward compatibility
    inverted_index = index_data['inverted_index']
    term_idf = index_data['idf_values']
    doc_vectors = index_data['doc_vectors']
    
    return inverted_index, term_idf, doc_vectors