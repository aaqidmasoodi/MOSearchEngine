# indexing.py
import os
import pickle
import time
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math
import nltk
import ssl
import parser

# Fix SSL certificate issue for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK resources with proper error handling
def download_nltk_resources():
    """Download required NLTK resources with error handling."""
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {e}")
            print("You may need to manually download NLTK resources.")

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

def build_inverted_index(documents):
    """Builds an enhanced inverted index with term frequencies from the documents."""
    print("Building inverted index...")
    start_time = time.time()
    
    inverted_index = defaultdict(dict)
    document_frequencies = defaultdict(int)
    term_frequencies = defaultdict(lambda: defaultdict(int))
    doc_lengths = {}
    
    # First pass: calculate term frequencies for each document
    for doc_id, doc in documents.items():
        # Weight title terms more than text terms
        title_terms = preprocess_text(doc["title"])
        text_terms = preprocess_text(doc["text"])
        
        # Combine terms with title terms having more weight
        all_terms = title_terms * 2 + text_terms  # Effectively gives title terms double weight
        doc_lengths[doc_id] = len(all_terms)
        
        # Count term frequencies in this document
        term_freq = defaultdict(int)
        for term in all_terms:
            term_freq[term] += 1
        
        # Record unique terms in this document
        for term, freq in term_freq.items():
            term_frequencies[doc_id][term] = freq
            document_frequencies[term] += 1
            inverted_index[term][doc_id] = freq
    
    # Calculate average document length
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
    
    # Calculate total number of documents
    N = len(documents)
    
    # Calculate IDF values
    idf_values = {}
    for term, df in document_frequencies.items():
        # BM25-style IDF formula
        idf_values[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    # Second pass: calculate normalized TF-IDF weights
    doc_vectors = {}
    for doc_id in documents:
        doc_vector = {}
        doc_length = doc_lengths.get(doc_id, 0)
        
        # Skip empty documents
        if doc_length == 0:
            continue
            
        for term, freq in term_frequencies[doc_id].items():
            if term in idf_values:
                # Calculate TF component with log normalization
                tf = 1 + math.log(freq) if freq > 0 else 0
                doc_vector[term] = tf * idf_values[term]
        
        # Normalize the document vector (L2 normalization)
        vector_magnitude = math.sqrt(sum(w**2 for w in doc_vector.values()))
        if vector_magnitude > 0:
            for term in doc_vector:
                doc_vector[term] /= vector_magnitude
                
        doc_vectors[doc_id] = doc_vector
    
    index_stats = {
        'num_documents': N,
        'num_terms': len(inverted_index),
        'avg_doc_length': avg_doc_length,
        'avg_terms_per_doc': sum(len(terms) for terms in term_frequencies.values()) / N if N > 0 else 0
    }
    
    print(f"Inverted index built in {time.time() - start_time:.2f} seconds")
    print(f"Index statistics: {index_stats}")
    
    return {
        'inverted_index': inverted_index,
        'doc_vectors': doc_vectors,
        'idf_values': idf_values,
        'doc_lengths': doc_lengths,
        'avg_doc_length': avg_doc_length,
        'term_frequencies': term_frequencies,
        'index_stats': index_stats
    }

def save_index(index_data, filename="index_data.pkl"):
    """Save the index data to disk for faster loading."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving index: {e}")
        return False

def load_index(filename="index_data.pkl"):
    """Load the index data from disk if available."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                index_data = pickle.load(f)
            print(f"Index loaded from {filename}")
            return index_data
        except Exception as e:
            print(f"Error loading index: {e}")
    
    return None

if __name__ == "__main__":
    # Check for existing index
    index_data = load_index()
    
    if not index_data:
        # Load parsed documents from parser.py
        print("Parsing documents...")
        documents = parser.parse_documents("datasets/cran.all.1400.xml")
        
        # Build inverted index
        index_data = build_inverted_index(documents)
        
        # Save index for future use
        save_index(index_data)
    
    # Print sample outputs
    inverted_index = index_data['inverted_index']
    sample_terms = list(inverted_index.keys())[:5]
    
    print("\nSample Inverted Index:")
    for term in sample_terms:
        doc_count = len(inverted_index[term])
        print(f"Term '{term}' appears in {doc_count} documents")
        sample_docs = list(inverted_index[term].items())[:3]
        for doc_id, freq in sample_docs:
            print(f"  - Doc {doc_id}: {freq} occurrences")
    
    # Print index statistics
    print("\nIndex Statistics:")
    for stat, value in index_data['index_stats'].items():
        print(f"{stat}: {value}")