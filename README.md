# Reddit Insights Chatbot with RAG
<p align="center">
  <img src="static/reddit_chatbot.png"/>
</p>

## Project Overview

This project extends the Reddit Insights Chatbot developed during Practicum I.
The goal is to build a fully automated, scalable, cloud-deployable Retrieval-Augmented Generation (RAG) system that:

- Collects Reddit posts programmatically

- Cleans and filters irrelevant content

- Performs semantic classification, sentiment scoring, and keyword extraction

- Indexes all final documents into Pinecone

- Provides an interactive chatbot that answers questions grounded ONLY in Reddit evidence

- Deploys to the cloud (Render.com) for public access

The chatbot supports three industries: Law, Construction, Tech

Project Structure
<pre>
project/
│   README.md
│   requirements.txt
│   run_pipeline.py
│   evaluate.py
│   data_collection.py
│   data_clean_and_classify.py
│   data_sentiment.py
│   store_index.py
│   reset_tracker.py
│   db.py
│   logger_utils.py
│   full_evaluation.ipynb
│
├── config/
│   ├── keywords.json
│   ├── questions.json
│   └── subreddits.json
│
├── data/           (Not included — too large, provided via Google Drive)
│   ├── reddit_data.csv
│   ├── reddit_data_clean.csv
│   ├── reddit_data_semantic_clean.csv
│   ├── reddit_data_sentiment.csv
│   └── evaluation_results.csv
│
├── templates/
│   chatbot.html
│   dashboard.html
│   manager.html
│   pipeline.html
│   base.html
│
├── static/
│   style.css
│   script.js
│   manager.js
│
└── render_app/     (Cloud version)
    ├── app.py
    ├── static/style.css
    ├──  Procfile
    ├── requirements.txt
    └── templates/chatbot.html

</pre>

# Installation & Setup

## STEP 01- Clone the repository:

Clone the repository

```bash
git clone https://github.com/jonishk/reddit-insights-app.git
cd reddit-insights-app
```

### STEP 02- Create and activate environment

```bash
conda create -n reditbot python=3.10
conda activate reditbot
````

### STEP 03- Install required packages
```bash
pip install -r requirements.txt
```
### STEP 04- Environment Variables (.env file)
Create a .env file (not included in repo):
```bash
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=reddit-insights
```

----------------------------------------------------------------------------------------------------

### Running the Full Pipeline
```bash
python run_pipeline.py
```
This executes:

1. data_collection.py

2. data_clean_and_classify.py

3. data_sentiment.py

4. store_index.py

----------------------------------------------------------------------------------------------------
### Running the Local Chatbot
```bash
python app.py
```
Then open in browser:
```bash
http://127.0.0.1:8080/
```
----------------------------------------------------------------------------------------------------
### Deployment (Render Version)
The Render version is in /render_app. It includes:

- Streamlined app.py
- Minimal requirements.txt
- Clean HTML/CSS chatbot interface

Live Render version:
```bash
https://reddit-insights-app.onrender.com/
```



