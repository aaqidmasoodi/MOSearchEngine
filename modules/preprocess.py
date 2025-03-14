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
    """
    Downloads the necessary NLTK resources:
    - 'punkt' for tokenization
    - 'stopwords' for filtering common words
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
    Preprocesses the input text:
    - Tokenizes the text into lowercase words
    - Removes stopwords, keeping specific domain-related words
    - Applies stemming only to longer words (length > 4)
    - Returns a cleaned list of words
    """
    # Set of stopwords, excluding domain-specific words like 'flow', 'wing', etc.
    stop_words = set(stopwords.words('english')) - {'flow', 'wing', 'speed', 'jet', 'gas'}
    
    # Initialize the Snowball Stemmer (lighter than the Porter stemmer)
    stemmer = SnowballStemmer('english')
    
    # Tokenize the text and convert all words to lowercase
    words = word_tokenize(text.lower())
    
    # Filter out non-alphanumeric words and stopwords, keeping only meaningful terms
    words = [w for w in words if (w.isalnum() and w not in stop_words) or (len(w) > 1 and w.isalpha())]
    
    # Apply stemming only to words that are longer than 4 characters
    words = [stemmer.stem(w) if len(w) > 4 else w for w in words]
    
    return words

def build_vocabulary(documents):
    """
    Builds a vocabulary (set of unique words) from all the documents.
    - Preprocesses both the title and the text of each document.
    - Returns a set of unique terms encountered in the documents.
    """
    vocabulary = set()
    for doc_id, doc in documents.items():
        # Preprocess title and text, and update the vocabulary with unique terms
        title_words = preprocess_text(doc["title"])
        text_words = preprocess_text(doc["text"])
        vocabulary.update(title_words)
        vocabulary.update(text_words)
    
    return vocabulary

def compute_doc_term_freqs(documents):
    """
    Precomputes term frequencies (TF) for all documents.
    - For each document, computes the frequency of terms in both the title and the text.
    - Returns a dictionary of term frequencies, document lengths, and average document length.
    """
    doc_term_freqs = {}  # Stores term frequencies for each document
    doc_lengths = {}  # Stores the number of terms in each document
    
    for doc_id, doc in documents.items():
        # Preprocess title and text of the document
        title_terms = preprocess_text(doc["title"])
        text_terms = preprocess_text(doc["text"])
        
        # Combine title and text terms, giving double weight to the title
        all_terms = title_terms * 2 + text_terms
        
        # Store the document's length (total number of terms)
        doc_lengths[doc_id] = len(all_terms)
        
        # Calculate term frequencies for the document
        term_freq = defaultdict(int)
        for term in all_terms:
            term_freq[term] += 1
            
        doc_term_freqs[doc_id] = term_freq  # Store the computed term frequencies for the document
    
    # Calculate the average document length
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
    
    return doc_term_freqs, doc_lengths, avg_doc_length

def preprocess_all(documents):
    """
    DEPRECATED: This function is replaced by 'build_inverted_index' from 'indexing.py'.
    - This function builds an inverted index, IDF values, and document vectors for all documents.
    - It issues a warning that it is deprecated.
    """
    from indexing import build_inverted_index
    import warnings
    warnings.warn("preprocess_all() is deprecated. Use build_inverted_index() instead.", DeprecationWarning)
    
    # Use the 'build_inverted_index' function from the 'indexing' module to generate the index data
    index_data = build_inverted_index(documents)
    
    # Return the inverted index, IDF values, and document vectors
    return index_data['inverted_index'], index_data['idf_values'], index_data['doc_vectors']
