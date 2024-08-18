import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def extract_text_from_pdf(pdf_path, start_page=14):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(start_page - 1, len(doc)):  # Adjusting for 0-based index
        page = doc.load_page(page_num)
        text += page.get_text()
    return text
bnss = extract_text_from_pdf("/IPC_latest.pdf")

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path, start_page=14):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(start_page - 1, len(doc)):  # Adjusting for 0-based index
        page = doc.load_page(page_num)
        text += page.get_text()
    return text
bnss = extract_text_from_pdf("/bnss.pdf")

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path, start_page=15):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(start_page - 1, len(doc)):  # Adjusting for 0-based index
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Extract text starting from page 14
bpc = extract_text_from_pdf("bpc_pdf.pdf")



import re


def preprocess_text(text):
    
  
     # Remove unnecessary white spaces
    text = re.sub(r'\s+', ' ', text)    
    # Repeal sections marked with "rep." or "[Repealed.]"
    text = re.sub(r'Section \d+.*?\s*rep\..*?(?=Section|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'Section \d+\.\s*\[Repealed\.\].*?(?=Section|$)', '', text, flags=re.DOTALL)    
    text = re.sub(r'\d+\*', '', text)
     # Remove the word 'illustration'
    text = re.sub(r'\billustration\b', '', text, flags=re.IGNORECASE)    
    # Remove square brackets but keep the content inside
    text = re.sub(r'\[(.*?)\]', r'\1', text)
    text = re.sub(r'\bSubs\b.*?(\.|\n)', '', text)
#     text = re.sub(r'\*', '', text)    
    # Replace terms
    text = re.sub(r'Code of Criminal Procedure \(Amendment\) Act \(\d+\)', '#', text)
    text = re.sub(r'Indian Penal Code', '$', text)    
    text = re.sub(r'-+', '', text)    
    text = re.sub(r'\d+', '', text)    
    return text



def preprocess_text_bnss(text):
    
  
     # Remove unnecessary white spaces
    text = re.sub(r'\s+', ' ', text)    
    # Repeal sections marked with "rep." or "[Repealed.]"
    text = re.sub(r'Section \d+.*?\s*rep\..*?(?=Section|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'Section \d+\.\s*\[Repealed\.\].*?(?=Section|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'\d+\*', '', text)    
    text = re.sub(r'\d+\.\s*', '', text)
    # Remove the word 'illustration'
    text = re.sub(r'\billustration\b', '', text, flags=re.IGNORECASE)    
#     # Remove square brackets but keep the content inside
#     text = re.sub(r'\[(.*?)\]', r'\1', text)    
    # Replace terms
    text = re.sub(r'Bharatiya Nagarik Suraksha Sanhita, 2023', '#', text)
    text = re.sub(r' Bharatiya Nyaya Sanhita, 2023', '$', text)    
    text = re.sub(r'-+', '', text)    
    text = re.sub(r'\d+', '', text)    
    return text



def preprocess_text_bpc(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bArt\.\s*\w*\b', '', text)
    text = re.sub(r'\d+', '', text)
    return text


ipc_preprocess = preprocess_text(ipc)
bnss_preprocess = preprocess_text_bnss(bnss)
bpc_preprocess = preprocess_text_bpc(bpc)


def tokenize_sentences(text):
    # Split text into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def find_matching_sentences(sentences1, sentences2):
    # Convert second list of sentences to a set for faster lookup
    sentences2_set = set(sentences2)    
    # Find matches and unmatched sentences
    matches = []
    unmatched = []
    for sentence in sentences1:
        if sentence in sentences2_set:
            matches.append(sentence)
        else:
            unmatched.append(sentence)    
    return matches, unmatched

def calculate_match_percentage(total_sentences, matched_sentences):
    if total_sentences == 0:
        return 0.0
    return (len(matched_sentences) / total_sentences) * 100



bnss_sentences = tokenize_sentences(bnss_preprocess)
ipc_sentences = tokenize_sentences(ipc_preprocess)
bpc_sentences = tokenize_sentences(bpc_preprocess)




def compute_cosine_similarity(doc1, doc2):
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")    
    # Fit and transform the documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    # Compute the cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]


similarity = compute_cosine_similarity(ipc_preprocess, bpc_preprocess)
print(f"Cosine Similarity: {similarity}")



def compute_cosine_similarity(doc1, doc2):
    # Initialize the TF-IDF Vectorizer
    vectorizer = CountVectorizer(stop_words="english")
    # Fit and transform the documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    # Compute the cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0][0]


similarity = compute_cosine_similarity(ipc_preprocess, bpc_preprocess)
print(f"Cosine Similarity: {similarity}")





def compute_cosine_similarity(doc1, doc2):
    # Initialize the TF-IDF Vectorizer
    vectorizer = CountVectorizer(stop_words="english")
    # Fit and transform the documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    # Compute the cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]) 
    return similarity_matrix[0][0]


similarity = compute_cosine_similarity(ipc_preprocess, bnss_preprocess)
print(f"Cosine Similarity: {similarity}")


