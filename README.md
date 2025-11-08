# Retrieval-Augmented Generation (RAG) – Document Question Answering App

**Overview**
This project is a simple RAG-based Question Answering system that lets us upload a PDF and ask questions about its content.
The app reads the document, finds the most relevant sections, and generates accurate answers — all using local models.
It’s useful for students, researchers, and professionals who want to quickly extract key insights from long reports, papers, or manuals.

**Features**
- Upload any PDF file.
- Automatically split and process text into chunks.
- Create embeddings using a local model (all-MiniLM-L6-v2).
- Store and search efficiently using FAISS vector database.
- Ask natural questions in the Streamlit app interface.
- Get summarized, context-aware answers instantly.

**Technologies Used**
- Python 3.12
- Streamlit – For the user interface
- PyPDF2 – To extract text from PDF files
- SentenceTransformers – For creating embeddings
- FAISS – For similarity search and retrieval
- Transformers (Hugging Face) – For language understanding

**Example Questions**(To be asked while testing, if we are testing on any research paper)
- What is the main idea of this paper?
- Which dataset was used in the experiment?
- Explain the workflow discussed in the document.

**Future Improvements**
- Add option to process multiple PDFs at once.
- Integrate a larger language model for better summarization.
- Save and reload previous vector indexes automatically.

Author

Syeda Nishat
