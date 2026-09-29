# Smart Document Q&A

## Problem

Finding a specific fact in several long PDF documents can take time, and answers without clear citations are difficult to verify.

## Solution

Smart Document Q&A lets you upload multiple PDFs, index their text locally, and ask questions in natural language. Gemini answers from the most relevant retrieved passages and displays the source filename and page number.

## RAG architecture

```text
PDF files
   │
   ▼
PyMuPDF page extraction ──► clean text ──► overlapping chunks + filename/page metadata
                                                │
                                                ▼
                                 Gemini embeddings (RETRIEVAL_DOCUMENT, 768 dims)
                                                │
                                                ▼
                                  normalized vectors in local FAISS index

Question ──► Gemini embedding (RETRIEVAL_QUERY, 768 dims) ──► top 5 FAISS matches
                                                                    │
                                                                    ▼
                                          retrieved context + question ──► Gemini 3.8 Flash
                                                                    │
                                                                    ▼
                                                    answer with source citations
```

## Features

- Upload and process multiple PDF files.
- Extract text page by page, preserving filename and page number.
- Clean and split text into overlapping chunks.
- Search a local FAISS CPU index with cosine-compatible normalized vectors.
- Ground answers in retrieved passages and show citations below every answer.
- Keep multiple questions and answers in Streamlit session state.
- Clear the indexed documents and chat with the reset control.
- Show processing progress and actionable messages for unreadable, empty, or scanned PDFs and API errors.

## Tech stack

- Python 3.13
- Streamlit
- PyMuPDF
- NumPy and FAISS CPU
- `google-genai` (Gemini Embeddings and Gemini 3.8 Flash)
- `python-dotenv`

## How the pipeline works

1. PyMuPDF extracts selectable text from each PDF page.
2. Text is cleaned and split into chunks of up to 1,200 characters with 200 characters of overlap. Each chunk retains its filename and page number.
3. `gemini-embedding-001` embeds document chunks with `RETRIEVAL_DOCUMENT` and 768 output dimensions. Vectors are L2-normalized and stored in an in-memory local FAISS inner-product index.
4. A question is embedded with `RETRIEVAL_QUERY`, then FAISS returns the five closest chunks.
5. Only the retrieved context and user's question are sent to `gemini-3.5-flash`. On transient errors, the app retries and then can fall back to `gemini-3.8-flash`. The prompt instructs the model to answer from the retrieved context and say when the answer is not found.
6. The answer and the retrieved chunks' filename/page citations appear in the chat history.

## Local setup

1. Install Python 3.13.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey). The app uses `gemini-embedding-001`, `gemini-3.5-flash` as the primary generation model, and `gemini-3.8-flash` as the fallback. Model availability and free-tier quotas depend on Google's current terms and account.
5. Copy `.env.example` to `.env` and replace the placeholder with your key:

   ```dotenv
   GEMINI_API_KEY=your_key_here
   ```

## Run the app

From the project directory, run:

```bash
streamlit run app.py
```

Upload PDFs in the sidebar and select **Process documents**. Then enter a question and select **Ask**.

## Security note

Keep your API key in `.env`; do not commit or share that file. The repository `.gitignore` excludes `.env`. The example file contains only a placeholder. Uploaded document content and questions are sent to the Gemini API for embeddings and answer generation.

## Future improvements

- Add OCR support for scanned PDFs.
- Provide configurable chunking and retrieval settings.
- Add document-level filters and downloadable citation reports.
- Persist indexes securely between sessions.
- Add evaluation examples to measure retrieval and answer quality.
