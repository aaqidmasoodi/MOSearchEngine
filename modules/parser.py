import xml.etree.ElementTree as ET
import re
import os

# File paths for Cranfield datasets
cran_docs_file = "datasets/cran.all.1400.xml"
cran_queries_file = "datasets/cran.qry.xml"
cran_qrels_file = "datasets/cranqrel.trec.txt"

def parse_documents(xml_file):
    """Parses the Cranfield document XML file with enhanced cleaning."""
    # Check if the provided XML file exists, and raise an error if not
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"Document file not found: {xml_file}")
    
    try:
        # Parse the XML file and get the root of the tree
        tree = ET.parse(xml_file)
        root = tree.getroot()
        documents = {}  # Dictionary to store parsed documents
        
        # Iterate over each <doc> element in the XML
        for doc in root.findall('doc'):
            # Extract document ID, title, author, bibliography, and text
            doc_id = doc.findtext("docno").strip() if doc.findtext("docno") else ""
            title = clean_text(doc.findtext("title").strip() if doc.findtext("title") else "")
            author = doc.findtext("author").strip() if doc.findtext("author") else ""
            bib = doc.findtext("bib").strip() if doc.findtext("bib") else ""
            text = clean_text(doc.findtext("text").strip() if doc.findtext("text") else "")
            
            # Add document to dictionary only if it has a valid ID
            if doc_id:  
                documents[doc_id] = {
                    "title": title,
                    "author": author,
                    "bib": bib,
                    "text": text
                }
        
        print(f"Parsed {len(documents)} documents from {xml_file}")
        return documents
    except ET.ParseError as e:
        # Handle any XML parsing errors and return an empty dictionary
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}

def parse_queries(xml_file):
    """Parses the Cranfield queries XML file with enhanced cleaning."""
    # Check if the queries XML file exists, and raise an error if not
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"Queries file not found: {xml_file}")
    
    try:
        # Parse the XML file and get the root of the tree
        tree = ET.parse(xml_file)
        root = tree.getroot()
        queries = {}  # Dictionary to store parsed queries
        
        # Iterate over each <top> element in the XML
        for top in root.findall("top"):
            # Extract query ID and text, ensuring they are valid
            query_id = top.findtext("num").strip() if top.findtext("num") else ""
            query_text = clean_text(top.findtext("title").strip() if top.findtext("title") else "")
            
            # Add query to dictionary if both query ID and text are valid
            if query_id and query_text:
                queries[query_id] = query_text
        
        print(f"Parsed {len(queries)} queries from {xml_file}")
        return queries
    except ET.ParseError as e:
        # Handle any XML parsing errors and return an empty dictionary
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}

def parse_qrels(qrels_file):
    """Parses the Cranfield relevance judgments file with validation."""
    # Check if the relevance judgments file exists
    if not os.path.exists(qrels_file):
        raise FileNotFoundError(f"Relevance judgments file not found: {qrels_file}")
    
    qrels = []  # List to store relevance judgments
    valid_count = 0  # Counter to track valid relevance judgments
    
    try:
        # Open and read the relevance judgments file
        with open(qrels_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Split each line into components: query ID, doc ID, and relevance
                parts = line.strip().split()
                
                # Validate if the line has the expected number of components (4)
                if len(parts) == 4:
                    query_id, _, doc_id, relevance = parts
                    try:
                        # Convert relevance to integer and add the judgment if valid
                        rel_value = int(relevance)
                        qrels.append((query_id, doc_id, rel_value))
                        valid_count += 1
                    except ValueError:
                        # Handle invalid relevance value
                        print(f"Warning: Invalid relevance value at line {line_num}: {relevance}")
                else:
                    # Handle malformed lines with incorrect number of components
                    print(f"Warning: Malformed line {line_num} in {qrels_file}: {line}")
        
        print(f"Parsed {valid_count} relevance judgments from {qrels_file}")
        return qrels
    except Exception as e:
        # Handle any errors during file reading
        print(f"Error reading qrels file {qrels_file}: {e}")
        return []

def clean_text(text):
    """Clean and normalize text, retaining numbers in context."""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep alphanumeric characters, spaces, and basic punctuation
    text = re.sub(r'[^\w\s\.,;:!?-]', '', text)
    
    # Normalize whitespace again after all replacements
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    try:
        # Parse documents, queries, and relevance judgments using the defined functions
        documents = parse_documents(cran_docs_file)
        print(f"Total documents: {len(documents)}")
        
        queries = parse_queries(cran_queries_file)
        print(f"Total queries: {len(queries)}")
        
        qrels = parse_qrels(cran_qrels_file)
        print(f"Total relevance judgments: {len(qrels)}")
        
        # Sample outputs to verify correctness
        if documents:
            print("\nSample Document (Doc 1):")
            print(f"Title: {documents['1']['title']}")
            print(f"Text (first 100 chars): {documents['1']['text'][:100]}...")
        
        if queries:
            print("\nSample Query (Query 1):")
            print(f"Text: {queries['1']}")
        
        if qrels:
            print("\nSample Relevance Judgments (first 5):")
            for qrel in qrels[:5]:
                print(f"Query {qrel[0]} - Doc {qrel[1]} - Relevance {qrel[2]}")
                
    except Exception as e:
        # Handle any errors that occur during parsing
        print(f"Error during parsing: {e}")
