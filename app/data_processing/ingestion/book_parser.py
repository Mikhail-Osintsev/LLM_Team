from typing import List               
from pypdf import PdfReader                     
import os                                 

def read_pdf(path: str) -> str:          
    r = PdfReader(path)               
    pages = [p.extract_text() or "" for p in r.pages]  
    return "\n".join(pages)              


def load_raw_texts(raw_dir: str) -> List[str]: 
    texts: List[str] = []               
    for name in sorted(os.listdir(raw_dir)): 
        p = os.path.join(raw_dir, name) 
        low = name.lower()           
        if low.endswith(".pdf"):         
            texts.append(read_pdf(p))   
    return texts                       