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
from modules import parser  # Updated import to use external parser module

# Disable SSL certificate verification (for NLTK resource downloading)
ssl._create_default_https_context = ssl._create_unverified_context

def download_nltk_resources():
    """
    Downloads necessary NLTK resources for text processing.
    Downloads the 'punkt' tokenizer and 'stopwords' corpus for text cleaning.
    """
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)  # Download resources silently
        except Exception as e:
            print(f"Warning: Failed to download NLTK resource '{resource}': {e}")
download_nltk_resources()

def preprocess_text(text):
    """
    Preprocesses the input text by:
    - Tokenizing the text into words
    - Removing stopwords (excluding specific domain-related words)
    - Stemming words (if length > 4) using SnowballStemmer
    - Returning a list of cleaned words
    """
    # Set of stopwords excluding domain-specific words ('flow', 'wing', etc.)
    stop_words = set(stopwords.words('english')) - {'flow', 'wing', 'speed', 'jet', 'gas'}
    
    # Initialize the Snowball Stemmer for stemming
    stemmer = SnowballStemmer('english')
    
    # Tokenize the input text into lowercase words
    words = word_tokenize(text.lower())
    
    # Filter out non-alphanumeric words, stopwords, and very short words
    words = [w for w in words if (w.isalnum() and w not in stop_words) or (len(w) > 1 and w.isalpha())]
    
    # Stem the words if their length is greater than 4
    words = [stemmer.stem(w) if len(w) > 4 else w for w in words]
    
    return words

def build_inverted_index(documents):
    """
    Builds the inverted index from the provided documents.
    - Calculates term frequencies (TF) and document frequencies (DF).
    - Computes TF-IDF (Term Frequency - Inverse Document Frequency) for each term in each document.
    - Normalizes the document vectors (for cosine similarity).
    - Returns a dictionary containing the inverted index, document vectors, IDF values, and statistics.
    """
    print("Building inverted index...")
    start_time = time.time()  # Start timer to measure index building time
    
    # Initialize data structures
    inverted_index = defaultdict(dict)  # Stores terms and their document frequencies
    document_frequencies = defaultdict(int)  # Tracks how many documents a term appears in
    term_frequencies = {}  # Stores term frequencies for each document
    doc_lengths = {}  # Stores the length of each document (number of terms)
    
    # Iterate over all documents to build the index
    for doc_id, doc in documents.items():
        # Preprocess title and text of the document
        title_terms = preprocess_text(doc["title"])
        text_terms = preprocess_text(doc["text"])
        
        # Combine title and text terms, give double weight to title terms
        all_terms = title_terms * 2 + text_terms
        
        # Store document length
        doc_lengths[doc_id] = len(all_terms)
        
        # Calculate term frequencies for the document
        term_freq = defaultdict(int)
        for term in all_terms:
            term_freq[term] += 1
        
        term_frequencies[doc_id] = dict(term_freq)  # Store term frequencies as regular dict
        
        # Update document frequency and inverted index
        for term, freq in term_freq.items():
            document_frequencies[term] += 1
            inverted_index[term][doc_id] = freq
    
    # Calculate average document length
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
    N = len(documents)  # Total number of documents
    
    # Calculate IDF values for each term
    idf_values = {term: math.log((N - df + 0.5) / (df + 0.5) + 1) 
                  for term, df in document_frequencies.items()}
    
    # Calculate document vectors (TF-IDF weighted vectors)
    doc_vectors = {}
    for doc_id in documents:
        doc_vector = {}
        doc_length = doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            continue  # Skip empty documents
        
        for term, freq in term_frequencies[doc_id].items():
            if term in idf_values:
                # Apply sublinear TF: tf = 1 + log(1 + freq)
                tf = 1 + math.log(1 + freq)
                doc_vector[term] = tf * idf_values[term]
        
        # Normalize the document vector (cosine normalization)
        vector_magnitude = math.sqrt(sum(w**2 for w in doc_vector.values()))
        if vector_magnitude > 0:
            for term in doc_vector:
                doc_vector[term] /= vector_magnitude
        doc_vectors[doc_id] = doc_vector
    
    # Calculate index statistics (document count, terms, average lengths)
    index_stats = {
        'num_documents': N,
        'num_terms': len(inverted_index),
        'avg_doc_length': avg_doc_length,
        'avg_terms_per_doc': sum(len(terms) for terms in term_frequencies.values()) / N if N > 0 else 0
    }
    
    print(f"Inverted index built in {time.time() - start_time:.2f} seconds")
    print(f"Index statistics: {index_stats}")
    
    # Return the complete index data
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
    """
    Saves the built index to a file using pickle.
    - The index is serialized and saved to a file.
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving index: {e}")
        return False

def load_index(filename="index_data.pkl"):
    """
    Loads a previously saved index from a file using pickle.
    - If the file exists, it will load the index data into memory.
    """
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
    # Main execution block
    index_data = load_index()  # Try loading existing index
    
    if not index_data:  # If no existing index, parse documents and build a new index
        print("Parsing documents...")
        documents = parser.parse_documents("datasets/cran.all.1400.xml")
        index_data = build_inverted_index(documents)
        save_index(index_data)  # Save the newly built index
    
    inverted_index = index_data['inverted_index']
    
    # Display a sample inverted index for the term 'wing'
    print("\nSample Inverted Index (term 'wing'):")
    if 'wing' in inverted_index:
        doc_count = len(inverted_index['wing'])
        print(f"Term 'wing' appears in {doc_count} documents")
        for doc_id, freq in list(inverted_index['wing'].items())[:3]:
            print(f"  - Doc {doc_id}: {freq} occurrences")
