import xml.etree.ElementTree as ET
import re
import os

# File paths
cran_docs_file = "datasets/cran.all.1400.xml"
cran_queries_file = "datasets/cran.qry.xml"
cran_qrels_file = "datasets/cranqrel.trec.txt"

def parse_documents(xml_file):
    """Parses the Cranfield document XML file with enhanced cleaning."""
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"Document file not found: {xml_file}")
        
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        documents = {}
        
        for doc in root.findall('doc'):
            doc_id = doc.findtext("docno").strip() if doc.findtext("docno") else ""
            title = clean_text(doc.findtext("title").strip() if doc.findtext("title") else "")
            author = doc.findtext("author").strip() if doc.findtext("author") else ""
            bib = doc.findtext("bib").strip() if doc.findtext("bib") else ""
            text = clean_text(doc.findtext("text").strip() if doc.findtext("text") else "")
            
            if doc_id:  # Only add document if it has a valid ID
                documents[doc_id] = {
                    "title": title,
                    "author": author,
                    "bib": bib,
                    "text": text
                }
        
        print(f"Parsed {len(documents)} documents from {xml_file}")
        return documents
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}

def parse_queries(xml_file):
    """Parses the Cranfield queries XML file with enhanced cleaning."""
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"Queries file not found: {xml_file}")
        
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        queries = {}
        
        for top in root.findall("top"):
            query_id = top.findtext("num").strip() if top.findtext("num") else ""
            query_text = clean_text(top.findtext("title").strip() if top.findtext("title") else "")
            
            if query_id and query_text:  # Only add queries with valid ID and text
                queries[query_id] = query_text
        
        print(f"Parsed {len(queries)} queries from {xml_file}")
        return queries
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}

def parse_qrels(qrels_file):
    """Parses the Cranfield relevance judgments file with validation."""
    if not os.path.exists(qrels_file):
        raise FileNotFoundError(f"Relevance judgments file not found: {qrels_file}")
        
    qrels = []
    valid_count = 0
    
    try:
        with open(qrels_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == 4:
                    query_id, _, doc_id, relevance = parts
                    try:
                        rel_value = int(relevance)
                        qrels.append((query_id, doc_id, rel_value))
                        valid_count += 1
                    except ValueError:
                        print(f"Warning: Invalid relevance value at line {line_num}: {relevance}")
                else:
                    print(f"Warning: Malformed line {line_num} in {qrels_file}: {line}")
        
        print(f"Parsed {valid_count} relevance judgments from {qrels_file}")
        return qrels
    except Exception as e:
        print(f"Error reading qrels file {qrels_file}: {e}")
        return []

def clean_text(text):
    """Clean and normalize text for better indexing."""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^\w\s\.,;:!?-]', '', text)
    
    # Remove isolated numbers (but keep numbers within words)
    text = re.sub(r'\b\d+\b', '', text)
    
    # Normalize whitespace again after all replacements
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    # Parse files with better error handling
    try:
        documents = parse_documents(cran_docs_file)
        print(f"Total documents: {len(documents)}")
        
        queries = parse_queries(cran_queries_file)
        print(f"Total queries: {len(queries)}")
        
        qrels = parse_qrels(cran_qrels_file)
        print(f"Total relevance judgments: {len(qrels)}")
        
        # Print sample outputs
        if documents:
            print("\nSample Document:")
            sample_doc_id = next(iter(documents))
            print(f"ID: {sample_doc_id}")
            print(f"Title: {documents[sample_doc_id]['title']}")
            print(f"Text (first 100 chars): {documents[sample_doc_id]['text'][:100]}...")
        
        if queries:
            print("\nSample Queries:")
            sample_queries = list(queries.items())[:2]
            for qid, qtext in sample_queries:
                print(f"Query {qid}: {qtext}")
        
        if qrels:
            print("\nSample Relevance Judgments:")
            for qrel in qrels[:5]:
                print(f"Query {qrel[0]} - Doc {qrel[1]} - Relevance {qrel[2]}")
                
    except Exception as e:
        print(f"Error during parsing: {e}")