# RAG (Retrieval-Augmented Generation) Project

A comprehensive Retrieval-Augmented Generation system designed to answer questions from financial reports using advanced natural language processing and vector search techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a RAG system that enables intelligent question-answering over financial documents. By combining information retrieval with generative AI, the system can provide accurate, contextually relevant answers to queries about financial reports and data.

RAG enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge sources before generating responses, ensuring answers are grounded in actual data rather than relying solely on the model's training data.

## ✨ Features

- **Document Processing**: Handles financial reports and extracts meaningful information
- **Semantic Search**: Uses vector embeddings for intelligent document retrieval
- **Question Answering**: Generates accurate answers based on retrieved context
- **Contextual Understanding**: Maintains context across financial documents
- **Scalable Architecture**: Designed to handle large document collections

## 📊 Dataset

This project uses the **Financial Reports QA Dataset** from Kaggle:

**Source**: [Data Retriever Dataset by ahmedsta](https://www.kaggle.com/datasets/ahmedsta/data-retreiver)

The dataset contains:
- Financial reports and documents
- Question-answer pairs for training and evaluation
- Structured financial data for RAG applications

### Dataset Setup

1. Download the dataset from the Kaggle link above
2. Place the dataset files in the `data/` directory
3. Ensure the data structure matches the expected format

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended) or `pip`
- Virtual environment support

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/eyasu11321238a/RAG.git
   cd RAG
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies using uv**
   ```bash
   # Install uv if you haven't already
   pip install uv
   
   # Install project dependencies
   uv add -r requirements.txt
   ```

   **Alternative: Using pip**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
RAG/
├── .venv/                  # Virtual environment
├── data/                   # Dataset directory
│   └── raw/               # Raw financial reports
├── src/                    # Source code
│   ├── embeddings/        # Vector embedding generation
│   ├── retrieval/         # Document retrieval logic
│   ├── generation/        # Answer generation
│   └── utils/             # Utility functions
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 💻 Usage

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline

# Initialize the RAG system
rag = RAGPipeline(
    data_path="data/raw/",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo"
)

# Process documents
rag.load_documents()
rag.create_embeddings()

# Ask a question
question = "What was the revenue growth in Q4?"
answer = rag.query(question)
print(answer)
```

### Running Scripts

```bash
# Process and index documents
python src/index_documents.py

# Run the question-answering system
python src/answer_questions.py --query "Your question here"

# Start interactive mode
python src/interactive.py
```

## 🛠 Technologies Used

- **Python 3.8+**: Core programming language
- **LangChain**: RAG framework and orchestration
- **Sentence Transformers**: Text embedding generation
- **FAISS/ChromaDB**: Vector database for similarity search
- **OpenAI API / Hugging Face**: Language model integration
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

## 🔄 How It Works

1. **Document Ingestion**
   - Financial reports are loaded and preprocessed
   - Documents are split into manageable chunks

2. **Embedding Generation**
   - Text chunks are converted to vector embeddings
   - Embeddings capture semantic meaning of the text

3. **Vector Storage**
   - Embeddings are stored in a vector database
   - Enables fast similarity search

4. **Query Processing**
   - User questions are converted to embeddings
   - Similar document chunks are retrieved

5. **Answer Generation**
   - Retrieved context is passed to the LLM
   - Model generates accurate, contextual answers

### RAG Pipeline Flow

```
User Query → Embedding → Vector Search → Context Retrieval → LLM → Answer
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv add -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [ahmedsta on Kaggle](https://www.kaggle.com/datasets/ahmedsta/data-retreiver)
- Inspired by RAG techniques from the research community
- Built with open-source libraries and tools

## 📧 Contact

**Project Maintainer**: Eyasu

**GitHub**: [eyasu11321238a](https://github.com/eyasu11321238a)

**Project Link**: [https://github.com/eyasu11321238a/RAG](https://github.com/eyasu11321238a/RAG)

---

⭐ If you find this project helpful, please consider giving it a star!

## 📚 Additional Resources

- [RAG Documentation](https://docs.langchain.com/docs/use-cases/question-answering)
- [Vector Databases Guide](https://www.pinecone.io/learn/vector-database/)
- [Financial NLP Resources](https://github.com/topics/financial-nlp)

## 🐛 Known Issues

- Check the [Issues](https://github.com/eyasu11321238a/RAG/issues) page for current bugs and feature requests

## 🗺 Roadmap

- [ ] Add support for more document types
- [ ] Implement advanced chunking strategies
- [ ] Add evaluation metrics dashboard
- [ ] Create web interface for easier interaction
- [ ] Support for multiple languages
- [ ] Integration with more LLM providers