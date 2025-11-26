# app.py
from src.search import RAGSearch

if __name__ == "__main__":
    rag = RAGSearch()

    query = "Who is CHAIRPERSON for SOCIAL TRANSFORMATION AND REMUNERATION COMMITTEE?"

    result = rag.search_advanced(
        query,
        top_k=5,
        min_score=0.1,
        return_context=True
    )

    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print("\n" + "=" * 80)
    print("SOURCES:")
    print("=" * 80)
    for i, source in enumerate(result["sources"], 1):
        print(f"\n{i}. Source: {source['source']}")
        print(f"   Page: {source['page']}")
        print(f"   Confidence: {source['score']:.4f}")
        print(f"   Preview: {source['preview'][:200]}...")
    
    print("\n" + "=" * 80)
    print(f"OVERALL CONFIDENCE: {result['confidence']:.4f}")
    print("=" * 80)