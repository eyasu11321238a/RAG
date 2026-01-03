from src.search import RAGSearch

# Initialize once (important for performance)
rag = RAGSearch()

def answer_question(query: str):
    result = rag.search_advanced(
        query,
        top_k=5,
        min_score=0.1,
        return_context=True
    )
    return result
