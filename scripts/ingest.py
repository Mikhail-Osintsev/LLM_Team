import os                                                      
from app.backend.core.config import get_settings               # настройки
from app.data_processing.ingestion.book_parser import load_raw_texts  # чтение сырья
from app.data_processing.ingestion.text_chunker import chunk_text     # чанкинг
from app.data_processing.indexing.index_builder import build_faiss    # построение индекса
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def main():
    s = get_settings()  
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    texts = load_raw_texts(s.RAW_DIR)  
    model = SentenceTransformer(s.EMBEDDING_MODEL)                         
    chunks = []                               
    for t in texts:
        chunks.extend(splitter.split_text(t)) 
    build_faiss(chunks, s.INDEX_DIR, model)                           

if __name__ == "__main__":                                    
    main()                                                   