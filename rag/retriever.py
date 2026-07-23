import faiss
import numpy as np
from rag.embedder import embed_text

class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.embeddings = embed_text(documents)
        
        # Normalize embeddings for Cosine Similarity
        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine Similarity
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=5):
        query_vec = embed_text([query])
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        retrieved_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])
        return retrieved_docs
