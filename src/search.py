# src/search.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir="faiss_store",
                 embedding_model="all-MiniLM-L6-v2",
                 llm_model="llama-3.1-8b-instant"):

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Auto-load or build vectorstore
        if not os.path.exists(f"{persist_dir}/faiss.index"):
            print("[DEBUG] Building new vectorstore...")
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            print("[DEBUG] Loading existing vectorstore...")
            self.vectorstore.load()

        groq_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_key, model_name=llm_model)

    def search_advanced(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
        return_context: bool = False,
    ):
        # 1. Query vector store
        raw_results = self.vectorstore.query(query, top_k)

        def to_similarity(dist):
            return 1.0 / (1.0 + float(dist))

        # 2. Filter by score
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

        # 3. Build context and formatted source metadata
        context_chunks = []
        source_list = []

        for r in results:
            meta = r["metadata"]
            text = meta.get("text", "")

            # Fixed: Now correctly accessing the metadata fields
            source_list.append({
                "source": meta.get("source_file"),
                "page": meta.get("page"),
                "score": r["similarity_score"],
                "preview": text[:300] + "..." if len(text) > 300 else text
            })

            context_chunks.append(text)

        full_context = "\n\n".join(context_chunks)
        confidence = max(r["similarity_score"] for r in results)

        # 4. Ask LLM
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