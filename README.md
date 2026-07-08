# AI Business Intelligence Assistant

An AI-powered BI assistant that lets you upload CSV, Excel, or PDF files and ask questions about your data in plain English.

## Features
- Upload CSV, Excel, or PDF files
- Ask questions in natural language — no SQL needed
- Auto-generates interactive charts using Plotly
- Data quality report (missing values, duplicates)
- Executive summary generation
- Chat history with follow-up question memory
- Export conversation as text file

## Tech Stack
- Python
- Streamlit
- LangChain
- ChromaDB
- OpenAI GPT-4o
- Pandas
- Plotly

## How to Run Locally

1. Clone the repo
2. Create a virtual environment and activate it
3. Run pip install -r requirements.txt
4. Create a .env file with your OPENAI_API_KEY
5. Run streamlit run app.py
