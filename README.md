
# 🛡️ Hate Speech Detection Model

This project is a text classification system that detects **Hate Speech**, **Offensive Language**, or **Neutral/Clean** text in user-generated content. It was developed as part of the Tienext Technical Task.

---

## 📊 Project Overview

The goal is to build a machine learning model that can classify text into one of three categories:
- **Hate Speech**
- **Offensive Language**
- **Neutral/Clean Language**

We implemented two different models:
- **Logistic Regression** (classical ML baseline)
- **BERT** (transformer-based deep learning model from Hugging Face)

---

## 📁 Dataset Description

The dataset used is based on [Davidson et al. (2017)](https://github.com/t-davidson/hate-speech-and-offensive-language), which contains tweets labeled into three classes:
- `0` — Hate Speech  
- `1` — Offensive Language  
- `2` — Neutral  

Each row contains:
- `tweet` (text)
- `class` (label)
- Other metadata columns

The data was cleaned, tokenized, and vectorized for model training.

---

## 🤖 Model Details

### ✅ Logistic Regression
- Used as a baseline with TF-IDF vectorization
- Quick, interpretable, and efficient on small datasets

### ✅ BERT (Bidirectional Encoder Representations from Transformers)
- Pre-trained: `bert-base-uncased` from Hugging Face
- Fine-tuned for 3-class classification
- Handles context, sarcasm, and semantic nuance much better

---

## ⚙️ Installation Requirements

To run the code, install the following Python packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn nltk transformers torch
```

Also, make sure to download NLTK stopwords and punkt tokenizer:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🚀 How to Run

1. Clone the repo or unzip the project folder
2. Place the dataset (`labeled_data.csv`) in the same directory
3. Run the script:

```bash
python hate_speech_detection_model.py
```

4. You can also test new sentences at the bottom of the script using:

```python
predict_text("I hate you!")
predict_text("You're a kind and wonderful person.")
```

---

## 🖼️ Sample Output Screenshot

![Sample Output](screenshot.png)  
*(Include a screenshot showing metrics, confusion matrix, and prediction sample)*

---

## ⚠️ Limitations & Future Work

- BERT is accurate but slower to train; not ideal for real-time applications without optimization
- Dataset may contain label noise and bias due to crowdsourced annotations
- Future improvements could include:
  - Better preprocessing (lemmatization, named entity removal)
  - Real-time inference API deployment (e.g., using FastAPI or Flask)
  - Adding more languages and multilingual models

---

## 📌 Task Context

This project was built for Tienext as a technical task under the supervision of:
> Subhadip Sarkar  
> CTO, Tienext

---

## 👨‍💻 Author

- [Jyoti]
- [GitHub : https://github.com/Jyoti-Hajjargi]
- [Linkdin : https://www.linkedin.com/in/jyoti-hajjargi-2070602bb]


