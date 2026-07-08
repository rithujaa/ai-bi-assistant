# AI Business Intelligence Assistant | GPT-4o + RAG + LangChain + ChromaDB

**Live App:** https://ai-bi-assistant.streamlit.app

An AI-powered business intelligence assistant that lets anyone upload a data file and ask questions about it in plain English. No SQL knowledge required. No coding required. Just upload and ask.

---

## What It Does

Upload a CSV, Excel, or PDF file and interact with your data through a conversational interface. The app uses GPT-4o to understand your questions, write real pandas code to compute answers, generate interactive charts, and explain findings in plain English.

---

## Features

### File Upload and Processing
- Supports CSV, Excel (.xlsx), and PDF files up to 200MB
- Automatically handles encoding issues (UTF-8, Latin-1) in CSV files
- Detects and lets you choose between multiple sheets in Excel files
- Extracts text from PDF documents for question answering
- Displays a data preview with the first 10 rows on upload

### AI-Powered Question Answering
- Ask any question in plain English about your data
- GPT-4o writes real pandas code to compute the exact answer from your dataset
- Self-healing code execution: if GPT-4o uses an unavailable library, the system automatically detects the error and asks GPT-4o to rewrite the code without it
- Answers are explained in clear, non-technical language for business users
- Follow-up questions work naturally using conversation history as context

### Automatic Chart Generation
- After answering each question, GPT-4o decides whether a chart would help visualize the result
- If yes, it automatically writes Plotly code and renders an interactive chart
- Charts are context-aware: bar charts for comparisons, histograms for distributions, line charts for trends

### Data Quality Report
- Automatically flags missing values per column with percentages
- Detects and reports duplicate rows
- Identifies constant columns with no variation
- Color-coded display: red for issues, green for clean checks

### Executive Summary
- One-click generation of a full AI-written executive summary of the dataset
- Covers what the dataset is about, key statistics, data quality issues, business insights, and recommended next steps
- Written by GPT-4o in a professional, business-friendly format

### Suggested Questions
- On upload, GPT-4o analyzes the dataset and suggests 4 specific, relevant questions to get started
- Questions are generated based on actual column names and data types
- Click any suggestion to auto-fill and answer instantly

### PDF Support with RAG
- PDF files are processed using Retrieval Augmented Generation (RAG)
- Text is extracted, split into chunks, converted into vector embeddings using OpenAI Embeddings, and stored in ChromaDB
- When you ask a question, the most semantically relevant chunks are retrieved and passed to GPT-4o as context
- Uses an ephemeral ChromaDB client so each PDF session is completely isolated with no data bleeding between uploads

### Chat History
- Full conversation history displayed in a chat interface
- Follow-up questions are aware of previous answers
- Clear conversation button to start fresh
- Export the entire conversation as a downloadable text file

### Settings and Customization
- Toggle to show or hide the pandas code used to answer each question
- Toggle to show or hide the raw computed result before explanation

---

## How the AI Works

This app uses several AI techniques working together:

**Code Generation and Execution**
For CSV and Excel files, the app does not search through text. Instead, it sends GPT-4o a description of the dataframe (column names, data types, sample rows) and asks it to write pandas Python code to answer the question. That code is then executed against the real dataset using Python's exec() function, producing a precise computed result. GPT-4o then explains the result in plain English.

**Retrieval Augmented Generation (RAG)**
For PDF files, the app uses RAG. The PDF text is split into overlapping chunks of around 1,000 characters using LangChain's RecursiveCharacterTextSplitter. Each chunk is converted into a 1,536-dimensional vector embedding using OpenAI's text-embedding-ada model via LangChain. These embeddings are stored in ChromaDB, an in-memory vector database. When you ask a question, the question is also embedded and compared against all stored chunk embeddings using cosine similarity. The four most semantically relevant chunks are retrieved and passed to GPT-4o as context to generate the answer.

**Self-Healing Code**
If GPT-4o generates code that uses a Python library not available in the environment, the app catches the ModuleNotFoundError, tells GPT-4o exactly which module failed, and asks it to rewrite the code using only available libraries (pandas, numpy, scipy). This happens automatically without any user intervention.

**Automatic Visualization**
After answering a question, the app sends the question and the answer to GPT-4o again with a separate prompt asking it to decide whether a chart would help, and if so, to write Plotly Express code to generate one. The chart is rendered inline in the chat.

---

## Tech Stack

| Category | Tools |
|---|---|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Database | ChromaDB (ephemeral, in-memory) |
| RAG Framework | LangChain |
| Data Processing | Pandas, NumPy, SciPy |
| Visualization | Plotly Express |
| PDF Parsing | PyPDF |
| Excel Parsing | OpenPyXL |
| Environment | Python 3.11, python-dotenv |

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/rithujaa/ai-bi-assistant.git
cd ai-bi-assistant
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create a .env file with your OpenAI API key**
```
OPENAI_API_KEY=your-key-here
```
**5. Run the app**
```bash
streamlit run app.py
```

The app will open at http://localhost:8501

---

## Project Structure
```
bi-assistant/
├── app.py              # Main application code
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes .env and venv from git
└── README.md           # This file
```
---

## Built By

Rithujaa Rajendrakumar  
NYU MS Data Science 2026  
https://www.linkedin.com/in/rithujaa/
