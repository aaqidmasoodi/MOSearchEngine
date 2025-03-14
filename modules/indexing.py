import os
import pickle
import time
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
import math
import nltk
import ssl
import parser

ssl._create_default_https_context = ssl._create_unverified_context

def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {e}")
download_nltk_resources()

def preprocess_text(text):
    stop_words = set(stopwords.words('english')) - {'flow', 'wing', 'speed', 'jet', 'gas'}
    stemmer = SnowballStemmer('english')
    words = word_tokenize(text.lower())
    words = [w for w in words if (w.isalnum() and w not in stop_words) or (len(w) > 1 and w.isalpha())]
    words = [stemmer.stem(w) if len(w) > 4 else w for w in words]
    return words

def build_inverted_index(documents):
    print("Building inverted index...")
    start_time = time.time()
    
    inverted_index = defaultdict(dict)
    document_frequencies = defaultdict(int)
    term_frequencies = {}  # Replace lambda with dict
    doc_lengths = {}
    
    for doc_id, doc in documents.items():
        title_terms = preprocess_text(doc["title"])
        text_terms = preprocess_text(doc["text"])
        all_terms = title_terms * 2 + text_terms
        doc_lengths[doc_id] = len(all_terms)
        
        term_freq = defaultdict(int)
        for term in all_terms:
            term_freq[term] += 1
        
        term_frequencies[doc_id] = dict(term_freq)  # Convert to regular dict
        for term, freq in term_freq.items():
            document_frequencies[term] += 1
            inverted_index[term][doc_id] = freq
    
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
    N = len(documents)
    
    idf_values = {term: math.log((N - df + 0.5) / (df + 0.5) + 1) 
                  for term, df in document_frequencies.items()}
    
    doc_vectors = {}
    for doc_id in documents:
        doc_vector = {}
        doc_length = doc_lengths.get(doc_id, 0)
        if doc_length == 0:
            continue
            
        for term, freq in term_frequencies[doc_id].items():
            if term in idf_values:
                tf = 1 + math.log(1 + freq)  # Sublinear TF
                doc_vector[term] = tf * idf_values[term]
        
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
    try:
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving index: {e}")
        return False

def load_index(filename="index_data.pkl"):
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
    index_data = load_index()
    if not index_data:
        print("Parsing documents...")
        documents = parser.parse_documents("datasets/cran.all.1400.xml")
        index_data = build_inverted_index(documents)
        save_index(index_data)
    
    inverted_index = index_data['inverted_index']
    print("\nSample Inverted Index (term 'wing'):")
    if 'wing' in inverted_index:
        doc_count = len(inverted_index['wing'])
        print(f"Term 'wing' appears in {doc_count} documents")
        for doc_id, freq in list(inverted_index['wing'].items())[:3]:
            print(f"  - Doc {doc_id}: {freq} occurrences")