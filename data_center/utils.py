import pickle
import pdfplumber

def loadPk(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = set(f.read().splitlines())
    return lines

def partial_match(source, text):
    i, n = 0, len(source)
    for c in text:
        while i < n and source[i] != c: i += 1
        if i >= n: return False
        i += 1
    return True

def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text