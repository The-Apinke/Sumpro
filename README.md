# SumPro

AI-powered document analysis tool that extracts insights from PDFs using RAG (Retrieval-Augmented Generation). Upload a document, choose your analysis mode, and get tailored summaries with follow-up Q&A.

**Live Demo:** [sumpro.streamlit.app](https://sumpro.streamlit.app) 

## Features

**Three Analysis Modes:**
- **Professional** - Extracts action items, decisions, and next steps from meetings and work documents
- **Tech Deep-Dive** - Breaks down technical concepts, implementation details, and architecture
- **Quick Digest** - Fast summaries with key points from any document

**Interactive Features:**
- Ask follow-up questions about your documents
- Explore specific sections in detail
- Generate follow-up emails (Professional mode)
- Extract key concepts (Tech mode)
- Conversation memory for multi-turn discussions

**Built-in Protections:**
- Rate limiting (2 analyses per user per day)
- Secure API key handling
- Error handling and validation

## How It Works

SumPro uses Retrieval-Augmented Generation instead of sending entire documents to an LLM:

1. **Extract** - Pull text from PDFs using PyMuPDF
2. **Chunk** - Split into 1000-character chunks with 200-character overlap
3. **Embed** - Convert chunks to vector embeddings with OpenAI
4. **Store** - Index in FAISS for fast similarity search
5. **Retrieve** - Find relevant chunks based on your query
6. **Generate** - Send context to GPT-4o-mini with mode-specific prompts

## Tech Stack

- **Streamlit** - Web interface
- **LangChain** - RAG framework
- **FAISS** - Vector similarity search
- **OpenAI** - Embeddings and LLM (GPT-4o-mini)
- **PyMuPDF** - PDF text extraction

## Running Locally

### Prerequisites
- Python 3.8+
- OpenAI API key


sumpro/
├── sumpro.py           # Main application
├── requirements.txt    # Python dependencies
├── .env.example       # API key template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Limitations

- PDF files only
- Best for documents under 50 pages
- Requires OpenAI API access
- Rate limited to 2 analyses per user per day
- No persistent storage (session-based)


## Author

Built by Apinke as a portfolio project demonstrating RAG implementation, prompt engineering, and production deployment.


