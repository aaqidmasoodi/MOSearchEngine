import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer  # Switch to Snowball for lighter stemming
import ssl
import nltk

# Fix SSL certificate issue for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

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
    """Tokenizes, filters stopwords, and stems words with domain awareness."""
    stop_words = set(stopwords.words('english')) - {'flow', 'wing', 'speed', 'jet', 'gas'}  # Keep aeronautical terms
    stemmer = SnowballStemmer('english')  # Lighter stemming than Porter
    
    # Convert to lowercase and tokenize
    words = word_tokenize(text.lower())
    
    # Filter words, keeping short technical terms
    words = [w for w in words if (w.isalnum() and w not in stop_words) or (len(w) > 1 and w.isalpha())]
    
    # Apply stemming selectively
    words = [stemmer.stem(w) if len(w) > 4 else w for w in words]  # Stem only longer words
    
    return words

def build_vocabulary(documents):
    """Builds a vocabulary from all documents."""
    vocabulary = set()
    for doc_id, doc in documents.items():
        title_words = preprocess_text(doc["title"])
        text_words = preprocess_text(doc["text"])
        vocabulary.update(title_words)
        vocabulary.update(text_words)
    return vocabulary

def compute_doc_term_freqs(documents):
    """Precomputes term frequencies for all documents."""
    doc_term_freqs = {}
    doc_lengths = {}
    
    for doc_id, doc in documents.items():
        title_terms = preprocess_text(doc["title"])
        text_terms = preprocess_text(doc["text"])
        
        all_terms = title_terms * 2 + text_terms  # Double weight for title
        doc_lengths[doc_id] = len(all_terms)
        
        term_freq = defaultdict(int)
        for term in all_terms:
            term_freq[term] += 1
            
        doc_term_freqs[doc_id] = term_freq
    
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
    return doc_term_freqs, doc_lengths, avg_doc_length

def preprocess_all(documents):
    """DEPRECATED: Use build_inverted_index from indexing.py instead."""
    from indexing import build_inverted_index
    import warnings
    warnings.warn("preprocess_all() is deprecated. Use build_inverted_index() instead.", DeprecationWarning)
    index_data = build_inverted_index(documents)
    return index_data['inverted_index'], index_data['idf_values'], index_data['doc_vectors']