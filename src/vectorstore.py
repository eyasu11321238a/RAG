import os
import faiss
import pickle
import numpy as np
from pathlib import Path
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(self, persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.embedding_pipeline = EmbeddingPipeline(embedding_model=embedding_model)
        self.index = None
        self.metadata = []
        self.processed_files = set()  # Track which files are already indexed

        os.makedirs(persist_dir, exist_ok=True)

    def exists(self):
        """Check if a valid vector store exists."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        files_path = os.path.join(self.persist_dir, "processed_files.pkl")
        
        return (os.path.exists(index_path) and 
                os.path.exists(metadata_path) and 
                os.path.exists(files_path))

    def build_from_documents(self, documents):
        """
        Build FAISS index from normalized documents.
        Expected format: [{"content": "...", "metadata": {...}}, ...]
        """
        print(f"[INFO] Building vectorstore from {len(documents)} documents")

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
        print(f"[INFO] Created {len(chunks)} chunks")

        # Generate embeddings
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        print(f"[INFO] Generated {len(embeddings)} embeddings")

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

        # Track processed files
        self.processed_files = {doc["metadata"]["source_file"] for doc in documents}

        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors")
        self.save()

    def add_documents(self, documents):
        """
        Add new documents to existing vector store incrementally.
        """
        if self.index is None:
            print("[WARNING] No existing index found. Building new one...")
            return self.build_from_documents(documents)

        # Filter out already processed files
        new_documents = [
            doc for doc in documents 
            if doc["metadata"]["source_file"] not in self.processed_files
        ]

        if not new_documents:
            print("[INFO] No new documents to add. All files already indexed.")
            return

        print(f"[INFO] Adding {len(new_documents)} new documents to vectorstore")

        # Convert to LangChain format
        from langchain_core.documents import Document
        lc_docs = []
        for doc in new_documents:
            lc_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                )
            )

        # Chunk and embed new documents
        chunks = self.embedding_pipeline.chunk_documents(lc_docs)
        print(f"[INFO] Created {len(chunks)} new chunks")

        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        print(f"[INFO] Generated {len(embeddings)} new embeddings")

        # Add to existing FAISS index
        self.index.add(np.array(embeddings).astype('float32'))

        # Append new metadata
        for chunk in chunks:
            self.metadata.append({
                "text": chunk.page_content,
                "source_file": chunk.metadata.get("source_file"),
                "page": chunk.metadata.get("page")
            })

        # Update processed files
        self.processed_files.update(
            doc["metadata"]["source_file"] for doc in new_documents
        )

        print(f"[INFO] Vector store now contains {self.index.ntotal} vectors")
        self.save()

    def rebuild(self, documents):
        """
        Force rebuild the entire vector store (useful if chunking/embedding settings changed).
        """
        print("[INFO] Force rebuilding vector store...")
        self.index = None
        self.metadata = []
        self.processed_files = set()
        self.build_from_documents(documents)

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
        """Save FAISS index, metadata, and processed files to disk."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        files_path = os.path.join(self.persist_dir, "processed_files.pkl")

        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        with open(files_path, "wb") as f:
            pickle.dump(self.processed_files, f)

        print(f"[INFO] Saved index to {index_path}")
        print(f"[INFO] Saved {len(self.metadata)} metadata entries")
        print(f"[INFO] Saved {len(self.processed_files)} processed file records")

    def load(self):
        """Load FAISS index, metadata, and processed files from disk."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        files_path = os.path.join(self.persist_dir, "processed_files.pkl")

        if not self.exists():
            raise FileNotFoundError(
                f"Complete vector store not found in {self.persist_dir}. "
                "Please build the index first."
            )

        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        with open(files_path, "rb") as f:
            self.processed_files = pickle.load(f)

        print(f"[INFO] Loaded index with {self.index.ntotal} vectors")
        print(f"[INFO] Loaded {len(self.metadata)} metadata entries")
        print(f"[INFO] Tracking {len(self.processed_files)} processed files")

    def get_stats(self):
        """Get statistics about the vector store."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_metadata": len(self.metadata),
            "processed_files": len(self.processed_files),
            "files": sorted(list(self.processed_files))
        }


# =============================================================================
# SAVE THE ABOVE AS: src/vectorstore.py
# =============================================================================


class RAGSearch:
    def __init__(self, persist_dir="faiss_store",
                 embedding_model="all-MiniLM-L6-v2",
                 llm_model="llama-3.1-8b-instant"):

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.data_dir = "data"

        # Smart initialization: only build if necessary
        self._initialize_vectorstore()

        groq_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_key, model_name=llm_model)

    def _initialize_vectorstore(self):
        """
        Initialize vector store intelligently:
        - Load if exists
        - Build if doesn't exist
        - Add new documents if some are missing
        """
        if self.vectorstore.exists():
            print("[INFO] Loading existing vectorstore...")
            self.vectorstore.load()
            
            # Check for new documents
            print("[INFO] Checking for new documents...")
            all_docs = load_all_documents(self.data_dir)
            self.vectorstore.add_documents(all_docs)
        else:
            print("[INFO] No existing vectorstore found. Building new one...")
            docs = load_all_documents(self.data_dir)
            self.vectorstore.build_from_documents(docs)

    def rebuild_index(self):
        """Force rebuild the entire vector store."""
        docs = load_all_documents(self.data_dir)
        self.vectorstore.rebuild(docs)

    def add_new_documents(self):
        """Scan data directory and add any new documents."""
        docs = load_all_documents(self.data_dir)
        self.vectorstore.add_documents(docs)

    def get_index_info(self):
        """Get information about the current vector store."""
        return self.vectorstore.get_stats()

    def search_advanced(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
        return_context: bool = False,
    ):
        # Query vector store
        raw_results = self.vectorstore.query(query, top_k)

        def to_similarity(dist):
            return 1.0 / (1.0 + float(dist))

        # Filter by score
        results = []
        for r in raw_results:
            sim = to_similarity(r["distance"])
            if sim >= min_score:
                r["similarity_score"] = sim
                results.append(r)

        if not results:
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "confidence": 0.0,
                "context": "" if return_context else None,
            }

        # Build context and formatted source metadata
        context_chunks = []
        source_list = []

        for r in results:
            meta = r["metadata"]
            text = meta.get("text", "")

            source_list.append({
                "source": meta.get("source_file"),
                "page": meta.get("page"),
                "score": r["similarity_score"],
                "preview": text[:300] + "..." if len(text) > 300 else text
            })

            context_chunks.append(text)

        full_context = "\n\n".join(context_chunks)
        confidence = max(r["similarity_score"] for r in results)

        # Ask LLM
        prompt = f"""Use the following context to answer the user's question concisely and accurately.

Context:
{full_context}

Question: {query}

Answer:"""

        response = self.llm.invoke(prompt)
        answer = response.content

        output = {
            "answer": answer,
            "sources": source_list,
            "confidence": confidence,
        }

        if return_context:
            output["context"] = full_context

        return output