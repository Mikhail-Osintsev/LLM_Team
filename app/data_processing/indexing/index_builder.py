import os                        
import pickle                          
import numpy as np                    
import faiss                                
from typing import List                    
from app.backend.services.embeddings import embed  
from sentence_transformers import SentenceTransformer

def build_faiss(chunks: list[str], index_dir: str, model: SentenceTransformer) -> None:
    os.makedirs(index_dir, exist_ok=True)
    vecs = embed(chunks, model).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "store.pkl"), "wb") as f:
        pickle.dump({"chunks": chunks}, f)