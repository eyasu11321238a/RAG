# src/search.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore  # ✅ FIXED: Import added
from src.data_loader import load_all_documents

load_dotenv()


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