# Char-Ngram Record Linker 🔗

A lightweight web app to match records between **two CSV files** using **character n-gram TF-IDF** and **cosine similarity**.

- Upload a *Left* CSV and a *Right* CSV
- Pick the **code/identifier** column for each side
- Select **one or more text columns** on each side — they’ll be **merged** into one field for matching
- Configure n-gram range and a minimum cosine threshold
- Download the full matches and also a **Paired Codes** CSV

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
