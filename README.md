# Record Linker (Cosine + Char n-gram)

Web app to match records between two CSV files using TF-IDF (char 3–5 n-grams) + cosine similarity.

## Features
- Upload two CSVs ("Left" and "Right")
- Choose which columns represent codes
- Select multiple columns to merge for each side into a comparable text field
- Tune n-gram range and cosine threshold
- Preview and download matched pairs (LeftCode, RightCode, LeftText, RightText, score)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
