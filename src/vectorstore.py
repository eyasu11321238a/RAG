import os
import faiss
import pickle
import numpy as np
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(self, persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.embedding_pipeline = EmbeddingPipeline(embedding_model=embedding_model)
        self.index = None
        self.metadata = []

        os.makedirs(persist_dir, exist_ok=True)

    def build_from_documents(self, documents):
        """
        Build FAISS index from normalized documents.
        Expected format: [{"content": "...", "metadata": {...}}, ...]
        """
        print(f"[DEBUG] Building vectorstore from {len(documents)} documents")

        # Convert normalized docs back to LangChain Document format for chunking
        from langchain_core.documents import Document
        lc_docs = []
        for doc in documents:
            lc_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
            )

        # Chunk documents
        chunks = self.embedding_pipeline.chunk_documents(lc_docs)
        print(f"[DEBUG] Created {len(chunks)} chunks")

        # Generate embeddings
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        print(f"[DEBUG] Generated {len(embeddings)} embeddings")

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        # Store metadata with text content
        self.metadata = []
        for chunk in chunks:
            self.metadata.append({
                "text": chunk.page_content,
                "source_file": chunk.metadata.get("source_file"),
                "page": chunk.metadata.get("page")
            })

        print(f"[DEBUG] FAISS index built with {self.index.ntotal} vectors")
        self.save()

    def query(self, query_text, top_k=5):
        """
        Query the FAISS index and return results with metadata.
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() or build_from_documents() first.")

        # Embed query
        query_embedding = self.embedding_pipeline.model.encode([query_text])

        # Search FAISS
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), top_k
        )

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "distance": float(distances[0][i]),
                    "metadata": self.metadata[idx]
                })

        return results

    def save(self):
        """Save FAISS index and metadata to disk."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[DEBUG] Saved index to {index_path}")
        print(f"[DEBUG] Saved metadata to {metadata_path}")

    def load(self):
        """Load FAISS index and metadata from disk."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Index or metadata not found in {self.persist_dir}")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[DEBUG] Loaded index with {self.index.ntotal} vectors")
        print(f"[DEBUG] Loaded {len(self.metadata)} metadata entries")s