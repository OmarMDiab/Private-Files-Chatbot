# Chatgbt Ghost: A private Chatgbt clone
Using huggingface and langchain this app offers: -
- **Model Switching:** Easily switch between different Huggingface models as per your requirements.
- **Document Chat:** Chat with your documents, including `(CSV, DOCX, PPTX, PDF)` files. Upload your documents and start querying them.
  
## Sample Run Videos
- [PDF Sample Run](https://drive.google.com/file/d/1pV-0HFy6f2Bj5J7dVN_MSCCbyjsbkS6i/view?usp=sharing)
- [CSV Sample Run](https://drive.google.com/file/d/11Zrc1USczjzVT6I5MQ9HQr68aot1Qr5i/view?usp=sharing)
  
## Easy Run ^^
In this Streamlit App you only just have to run the python script **`without`** the need of this code: `streamlit run script.py`

Just run: -
```bash
python Chatgbt_Ghost.py
```

## Models Used
- hkunlp/instructor-xl (For Embeddings)
- google/flan-t5-xxl
- google/flan-t5-small
- google/flan-t5-base
- mistral-7b-instruct-v0.2.Q4_K_S (For local runs using LlamaCpp)
