from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    def __init__(self, embedding_model="all-MiniLM-L6-v2",
                 chunk_size=1000, chunk_overlap=200):

        self.model = SentenceTransformer(embedding_model)
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def chunk_documents(self, documents):
        """
        Preserve metadata: source_file + page
        """
        chunks = []

        for doc in documents:
            text_chunks = self.chunker.split_text(doc.page_content)

            for chunk_text in text_chunks:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "source_file": doc.metadata.get("source_file"),
                            "page": doc.metadata.get("page")
                        }
                    )
                )

        return chunks

    def embed_chunks(self, chunks):
        texts = [c.page_content for c in chunks]
        return self.model.encode(texts, show_progress_bar=True)