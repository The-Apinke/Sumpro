# SumPro

AI-powered document analysis tool that extracts insights from PDFs using RAG (Retrieval-Augmented Generation). Upload a document, choose your analysis mode, and get tailored summaries with follow-up Q&A.

**Live Demo:** [sumpro.streamlit.app](https://sumpro.streamlit.app) *(replace with your actual URL after deployment)*

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

### Setup

```bash
# Clone the repo
git clone https://github.com/your-username/sumpro.git
cd sumpro

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run
streamlit run sumpro.py
```

Open `http://localhost:8501` in your browser.

## Deploying to Streamlit Cloud

1. Fork or clone this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo and branch
5. Set main file to `sumpro.py`
6. Click "Advanced settings" → "Secrets"
7. Add your OpenAI API key:
```
OPENAI_API_KEY = "sk-your-key-here"
```
8. Click "Deploy"

Your app will be live at `your-app-name.streamlit.app` in 2-3 minutes.

## API Costs

This app uses OpenAI's API:
- Embeddings: ~$0.0001 per 1K tokens
- GPT-4o-mini: ~$0.0001 per 1K input, ~$0.0003 per 1K output
- Average analysis: $0.002-0.01 per document

Rate limiting (2 per user per day) helps control costs for public deployment.

## Project Structure

```
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

## Future Enhancements

- Support for Word docs, images, web pages
- Document comparison mode
- Export summaries to PDF/Word
- Local LLM support (eliminate API costs)
- User authentication for higher limits

## License

MIT License - feel free to use this for your own projects.

## Author

Built by Apinke as a portfolio project demonstrating RAG implementation, prompt engineering, and production deployment.

## Contact

- Portfolio: [your-portfolio-url]
- LinkedIn: [your-linkedin]
- GitHub: [@your-username](https://github.com/your-username)
