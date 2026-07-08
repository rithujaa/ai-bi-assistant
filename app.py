import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from openai import OpenAI
import plotly.express as px
import chromadb
import io

# ── SETUP ─────────────────────────────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="AI BI Assistant", layout="wide", page_icon="🤖")

# ── CUSTOM CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: #a8b2d8;
        margin: 5px 0 0 0;
        font-size: 0.95rem;
    }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 15px 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #e94560;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #a8b2d8;
        margin-top: 4px;
    }
    .suggestion-btn {
        background: #1e1e2e;
        border: 1px solid #e94560;
        border-radius: 8px;
        padding: 8px 12px;
        color: #e94560;
        cursor: pointer;
        font-size: 0.85rem;
        margin: 4px;
    }
    .warning-card {
        background: #2d1f1f;
        border-left: 4px solid #e94560;
        border-radius: 6px;
        padding: 10px 15px;
        margin: 5px 0;
        font-size: 0.85rem;
        color: #ffb3b3;
    }
    .good-card {
        background: #1f2d1f;
        border-left: 4px solid #4caf50;
        border-radius: 6px;
        padding: 10px 15px;
        margin: 5px 0;
        font-size: 0.85rem;
        color: #b3ffb3;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Business Intelligence Assistant</h1>
    <p>Upload a CSV, Excel, or PDF file and ask questions about your data using natural language.</p>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "executive_summary" not in st.session_state:
    st.session_state.executive_summary = ""

# ── HELPER FUNCTIONS ──────────────────────────────────────────────

def load_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    df = None
    raw_text = ""

    if file_type == "csv":
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")

    elif file_type == "xlsx":
        try:
            xl = pd.ExcelFile(uploaded_file)
            sheet_names = xl.sheet_names
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("Multiple sheets found. Pick one:", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
            else:
                df = pd.read_excel(uploaded_file, sheet_name=0)
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            return None, "", file_type

    elif file_type == "pdf":
        try:
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                st.error("This PDF is password protected.")
                return None, "", file_type
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text + "\n"
            if len(raw_text.strip()) == 0:
                st.error("Could not extract text — this may be a scanned PDF.")
                return None, "", file_type
        except Exception as e:
            st.error(f"Could not read PDF: {e}")
            return None, "", file_type

    return df, raw_text, file_type


def build_qa_chain(text, collection_name="default"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    chroma_client = chromadb.EphemeralClient()
    vectorstore = Chroma.from_texts(
        chunks,
        embeddings,
        client=chroma_client,
        collection_name=collection_name
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )
    return qa_chain


def answer_data_question(question, df, chat_history):
    history_text = ""
    if chat_history:
        history_text = "Previous questions and answers:\n"
        for item in chat_history[-3:]:
            history_text += f"Q: {item['question']}\nA: {item['answer']}\n\n"

    df_info = f"""
The dataframe is called 'df'. It has {df.shape[0]} rows and {df.shape[1]} columns.
Columns: {', '.join(df.columns.tolist())}
Data types:
{df.dtypes.to_string()}
First 5 rows:
{df.head().to_string()}
"""

    prompt = f"""
You are a data analyst. You have access to a pandas dataframe called 'df'.

Here is information about the dataframe:
{df_info}

{history_text}

The user asked: "{question}"

Write a single Python code block using pandas to answer this question.
The result should be stored in a variable called 'result'.
Only write the code — no explanation, no markdown, just plain Python code.
Only use these available modules: pandas (pd), numpy (np), scipy (scipy).
Do not use any other modules — especially not matplotlib, seaborn, statsmodels, sklearn, or nltk.
Do not import anything — all modules are already available as pd, np, and scipy.
The 'result' variable should contain a number, string, dataframe, or series — not a plot.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    code = response.choices[0].message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    try:
        import scipy
        import numpy as np
        local_vars = {"df": df, "pd": pd, "scipy": scipy, "np": np}
        exec(code, {}, local_vars)
        result = local_vars.get("result", "No result returned.")

        explanation_prompt = f"""
The user asked: "{question}"
The pandas code returned this result: {result}
Explain this result clearly and concisely in plain English for a business user.
Keep it to 2-3 sentences maximum.
"""
        explanation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0
        )
        explanation = explanation_response.choices[0].message.content.strip()
        return explanation, code, result

    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
        retry_prompt = f"""
The following code failed because the module '{missing_module}' is not available:

{code}

Rewrite this code to answer the same question without using '{missing_module}'.
Only use pandas (pd), numpy (np), or scipy (scipy).
Store the result in a variable called 'result'.
Only write the code — no explanation, no markdown.
"""
        retry_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": retry_prompt}],
            temperature=0
        )
        retry_code = retry_response.choices[0].message.content.strip()
        retry_code = retry_code.replace("```python", "").replace("```", "").strip()

        try:
            import scipy
            import numpy as np
            local_vars = {"df": df, "pd": pd, "scipy": scipy, "np": np}
            exec(retry_code, {}, local_vars)
            result = local_vars.get("result", "No result returned.")

            explanation_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"""
The user asked: "{question}"
The code returned: {result}
Explain this clearly in 2-3 sentences for a business user.
"""}],
                temperature=0
            )
            return explanation_response.choices[0].message.content.strip(), retry_code, result

        except Exception as e2:
            return f"I had trouble computing that. Try rephrasing your question. Error: {e2}", retry_code, None

    except Exception as e:
        return f"I had trouble computing that. Try rephrasing your question. Error: {e}", code, None


def generate_chart(question, df, answer):
    df_info = f"""
Columns: {', '.join(df.columns.tolist())}
Data types:
{df.dtypes.to_string()}
"""

    prompt = f"""
You are a data visualization expert. You have a pandas dataframe called 'df'.

Dataframe info:
{df_info}

The user asked: "{question}"
The answer was: "{answer}"

Decide if a chart would help visualize this answer.
If yes, write Python code using plotly.express to create a chart. Store it in a variable called 'fig'.
If no chart is needed (e.g. for simple single-number answers), just write: fig = None

Only write the code — no explanation, no markdown, just plain Python code.
Only use plotly.express (px) for charts. Do not use matplotlib or seaborn.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    code = response.choices[0].message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    try:
        local_vars = {"df": df, "pd": pd, "px": px}
        exec(code, {}, local_vars)
        fig = local_vars.get("fig", None)
        return fig, code
    except Exception as e:
        return None, f"Chart error: {e}"


def generate_executive_summary(df):
    df_info = f"""
Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
Columns: {', '.join(df.columns.tolist())}
Basic statistics:
{df.describe(include='all').to_string()}
Missing values per column:
{df.isnull().sum().to_string()}
Sample data (first 10 rows):
{df.head(10).to_string()}
"""

    prompt = f"""
You are a senior business analyst. Analyze this dataset and write a concise executive summary.

Dataset information:
{df_info}

Write an executive summary that includes:
1. What this dataset appears to be about
2. Key statistics and highlights
3. Any data quality issues (missing values, outliers)
4. 3 key business insights from the data
5. Recommended next steps for analysis

Keep it professional, clear, and actionable. Use bullet points where appropriate.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def generate_suggested_questions(df):
    df_info = f"""
Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
Columns: {', '.join(df.columns.tolist())}
Data types:
{df.dtypes.to_string()}
First 3 rows:
{df.head(3).to_string()}
"""

    prompt = f"""
You are a data analyst. Based on this dataset, suggest 4 interesting questions a business user might ask.
Make them specific to the actual column names in the dataset.
Return only the 4 questions, one per line, no numbering, no extra text.

Dataset info:
{df_info}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    questions = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in questions if q.strip()][:4]


def get_data_quality_report(df):
    issues = []
    good = []

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = round(count / len(df) * 100, 1)
            issues.append(f"⚠️ '{col}' has {count} missing values ({pct}%)")
    else:
        good.append("✅ No missing values found")

    # Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append(f"⚠️ {dupes} duplicate rows found")
    else:
        good.append("✅ No duplicate rows")

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues.append(f"⚠️ Constant columns (no variation): {', '.join(constant_cols)}")
    else:
        good.append("✅ All columns have variation")

    return issues, good


def export_chat_history(chat_history):
    """Export chat history as a text file"""
    output = "AI BI Assistant — Conversation Export\n"
    output += "=" * 50 + "\n\n"
    for i, item in enumerate(chat_history, 1):
        output += f"Q{i}: {item['question']}\n"
        output += f"A{i}: {item['answer']}\n"
        output += "-" * 30 + "\n"
    return output


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Upload Your File")
    uploaded_file = st.file_uploader(
        "Supports CSV, Excel, PDF",
        type=["csv", "xlsx", "pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Size:** {round(uploaded_file.size / 1024, 1)} KB")

    st.divider()
    st.markdown("### 💡 How to use")
    st.markdown("""
- Upload any **CSV**, **Excel**, or **PDF** file
- View automatic **data quality report**
- Click a **suggested question** to get started
- Ask **follow-up questions** naturally
- Generate an **executive summary**
- **Export** your conversation
""")

    st.divider()
    st.markdown("### ⚙️ Settings")
    show_code = st.toggle("Show pandas code", value=True)
    show_raw = st.toggle("Show raw results", value=False)

# ── MAIN AREA ─────────────────────────────────────────────────────

if uploaded_file is not None:

    # Reset session when a new file is uploaded
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.df = None
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.session_state.file_name = uploaded_file.name
        st.session_state.file_type = None
        st.session_state.suggested_questions = []
        st.session_state.summary_generated = False
        st.session_state.executive_summary = ""

    # Load file only once
    if st.session_state.df is None and st.session_state.qa_chain is None:
        with st.spinner("Reading your file..."):
            df, raw_text, file_type = load_file(uploaded_file)
            st.session_state.file_type = file_type

            if df is not None:
                st.session_state.df = df
                # Generate suggested questions in background
                with st.spinner("Generating suggested questions..."):
                    st.session_state.suggested_questions = generate_suggested_questions(df)

            elif raw_text:
                with st.spinner("Building AI knowledge base from PDF..."):
                    collection_name = uploaded_file.name.replace(".", "_").replace(" ", "_").lower()
                    st.session_state.qa_chain = build_qa_chain(raw_text, collection_name)

    # ── TABULAR DATA (CSV/XLSX) ────────────────────────────────────
    if st.session_state.df is not None:
        df = st.session_state.df

        # Metric cards
        missing_count = df.isnull().sum().sum()
        dupes_count = df.duplicated().sum()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{df.shape[0]:,}</div>
                <div class="label">Total Rows</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{df.shape[1]}</div>
                <div class="label">Columns</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{missing_count:,}</div>
                <div class="label">Missing Values</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{dupes_count:,}</div>
                <div class="label">Duplicate Rows</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for different views
        tab1, tab2 = st.tabs(["📊 Data Preview", "🔍 Data Quality"])

        with tab1:
            st.dataframe(df.head(10), use_container_width=True)

        with tab2:
            issues, good = get_data_quality_report(df)
            if issues:
                st.markdown("**Issues Found:**")
                for issue in issues:
                    st.markdown(f'<div class="warning-card">{issue}</div>', unsafe_allow_html=True)
            if good:
                st.markdown("**Looks Good:**")
                for g in good:
                    st.markdown(f'<div class="good-card">{g}</div>', unsafe_allow_html=True)

        # Executive Summary outside tabs so button click doesn't reset tab
        st.markdown("### 📋 Executive Summary")
        if not st.session_state.summary_generated:
            if st.button("Generate Executive Summary", type="primary"):
                with st.spinner("Generating executive summary..."):
                    st.session_state.executive_summary = generate_executive_summary(df)
                    st.session_state.summary_generated = True

        if st.session_state.summary_generated:
            st.markdown(st.session_state.executive_summary)
            if st.button("Regenerate"):
                st.session_state.summary_generated = False
                st.session_state.executive_summary = ""

        # Suggested questions
        if st.session_state.suggested_questions:
            st.markdown("### 💡 Suggested Questions")
            cols = st.columns(2)
            for i, q in enumerate(st.session_state.suggested_questions):
                with cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state.pending_question = q

    # ── PDF ───────────────────────────────────────────────────────
    elif st.session_state.qa_chain is not None:
        st.success("📄 PDF uploaded! Ready to answer questions.")

    else:
        st.stop()

    # ── CHAT INTERFACE ────────────────────────────────────────────
    st.divider()

    # Export button
    if st.session_state.chat_history:
        col_export, col_clear = st.columns([1, 1])
        with col_export:
            export_text = export_chat_history(st.session_state.chat_history)
            st.download_button(
                label="📥 Export Conversation",
                data=export_text,
                file_name="conversation_export.txt",
                mime="text/plain"
            )
        with col_clear:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # Show chat history
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            if item.get("chart") is not None:
                st.plotly_chart(item["chart"], use_container_width=True)

    # Handle suggested question clicks
    pending = st.session_state.get("pending_question", None)

    # Question input
    question = st.chat_input("Ask anything about your data...")

    # Use pending question from suggested button if exists
    if pending and not question:
        question = pending
        st.session_state.pending_question = None

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                if st.session_state.df is not None:
                    answer, code_used, raw_result = answer_data_question(
                        question,
                        st.session_state.df,
                        st.session_state.chat_history
                    )
                    st.write(answer)

                    fig, chart_code = generate_chart(question, st.session_state.df, answer)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

                    if show_code:
                        with st.expander("🔍 See pandas code"):
                            st.code(code_used, language="python")

                    if show_raw and raw_result is not None:
                        with st.expander("📊 See raw result"):
                            st.write(raw_result)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "chart": fig
                    })

                else:
                    result = st.session_state.qa_chain.invoke({"query": question})
                    answer = result["result"]
                    st.write(answer)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "chart": None
                    })

else:
    # Landing page when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2 style="color: #a8b2d8;">👈 Upload a file to get started</h2>
        <p style="color: #6b7280; font-size: 1rem;">
            Supports CSV, Excel (.xlsx), and PDF files up to 200MB
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">📊</div>
            <div style="color: #e94560; font-weight: bold; margin-top: 8px;">Ask Questions</div>
            <div style="color: #a8b2d8; font-size: 0.85rem; margin-top: 4px;">
                Ask anything in plain English (no SQL needed)
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">📈</div>
            <div style="color: #e94560; font-weight: bold; margin-top: 8px;">Auto Charts</div>
            <div style="color: #a8b2d8; font-size: 0.85rem; margin-top: 4px;">
                Automatically generates interactive visualizations
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem;">🔍</div>
            <div style="color: #e94560; font-weight: bold; margin-top: 8px;">Data Quality</div>
            <div style="color: #a8b2d8; font-size: 0.85rem; margin-top: 4px;">
                Instant data quality report and executive summary
            </div>
        </div>""", unsafe_allow_html=True)