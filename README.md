# Medical-Chatbot

## 📌 Project Overview

This project builds and evaluates a medical chatbot using two approaches: Retrieval-Augmented Generation (RAG) and LLM-only. The RAG chatbot grounds its responses in the Gale Encyclopedia of Medicine, while the LLM-only chatbot generates answers without external retrieval.

The systems are compared across three key metrics:

- Accuracy – correctness of answers to medical questions

- Faithfulness – alignment with retrieved sources

- Privacy – safe handling of out-of-scope queries

Findings show that RAG improves reliability and safety by reducing hallucinations and declining to answer out-of-scope questions, whereas LLM-only models are more fluent but risk producing ungrounded or speculative outputs.

✅ Final Practicum Submission for **MSDS696 – Data Science Practicum II**  
🎓 Regis University  
👨‍💻 Author: Jonish Bishwakarma

## 📂 Dataset

Dataset used: (https://staibabussalamsula.ac.id/wp-content/uploads/2024/06/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf)


# How to run?

# STEPS:

Clone the repository

```bash
git clone https://github.com/jonishk/Medibot

```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n reditbot python=3.10.18 -y

```bash
conda activate medibot
````
### STEP 02- install requirements
```bash
pip install -r requirements.txt
```

### STEP 04 - run the app using app.py
```bash
python app.py
```
### STEP 05 - on your browser run:
```bash
localhost:8080

