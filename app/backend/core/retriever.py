from app.backend.services.embeddings import embed     
from app.backend.services.vector_store import store  

def retrieve(query: str, top_k: int = 4):
    qv = embed([query])      
    return store.search(qv.astype("float32"), k=top_k)