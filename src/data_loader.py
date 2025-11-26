from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


# ---------------------------------------------------------
#  Metadata Normalization Helper
# ---------------------------------------------------------
def normalize_metadata(docs, source_file):
    """
    Convert LangChain Document objects into a consistent dict format:
    {
        "content": "...",
        "metadata": {
            "source_file": "filename.pdf",
            "page": 1
        }
    }
    """
    normalized = []
    for d in docs:
        normalized.append({
            "content": d.page_content,
            "metadata": {
                "source_file": source_file,
                "page": d.metadata.get("page", 1)
            }
        })
    return normalized


# ---------------------------------------------------------
#  Load ALL documents from /data
# ---------------------------------------------------------
def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported file types from the data directory and convert
    to a normalized document format for vector storage.
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    # -----------------------
    # PDF Files
    # -----------------------
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            docs = normalize_metadata(loaded, pdf_file.name)
            documents.extend(docs)
            print(f"[DEBUG] Loaded {len(docs)} PDF pages")
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # -----------------------
    # TXT Files
    # -----------------------
    txt_files = list(data_path.glob("**/*.txt"))
    print(f"[DEBUG] Found {len(txt_files)} TXT files")

    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            docs = normalize_metadata(loaded, txt_file.name)
            documents.extend(docs)
            print(f"[DEBUG] Loaded {len(docs)} TXT segments")
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # -----------------------
    # CSV Files
    # -----------------------
    # CSV Files
    csv_files = list(data_path.glob("**/*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            loaded = None
            
            for encoding in encodings:
                try:
                    loader = CSVLoader(
                        str(csv_file),
                        encoding=encoding,
                        csv_args={
                            'delimiter': ',',
                            'quotechar': '"',
                        }
                    )
                    loaded = loader.load()
                    print(f"[DEBUG] Successfully loaded CSV with {encoding} encoding")
                    break
                except Exception as e:
                    continue
            
            if loaded:
                docs = normalize_metadata(loaded, csv_file.name)
                documents.extend(docs)
                print(f"[DEBUG] Loaded {len(docs)} CSV rows")
            else:
                print(f"[WARNING] Could not load CSV {csv_file.name} with any encoding")
                
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # -----------------------
    # Excel Files
    # -----------------------
    xlsx_files = list(data_path.glob("**/*.xlsx"))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files")

    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            docs = normalize_metadata(loaded, xlsx_file.name)
            documents.extend(docs)
            print(f"[DEBUG] Loaded {len(docs)} Excel entries")
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # -----------------------
    # DOCX Files
    # -----------------------
    docx_files = list(data_path.glob("**/*.docx"))
    print(f"[DEBUG] Found {len(docx_files)} Word files")

    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            docs = normalize_metadata(loaded, docx_file.name)
            documents.extend(docs)
            print(f"[DEBUG] Loaded {len(docs)} DOCX segments")
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # -----------------------
    # JSON Files
    # -----------------------
    json_files = list(data_path.glob("**/*.json"))
    print(f"[DEBUG] Found {len(json_files)} JSON files")

    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            docs = normalize_metadata(loaded, json_file.name)
            documents.extend(docs)
            print(f"[DEBUG] Loaded {len(docs)} JSON entries")
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents


