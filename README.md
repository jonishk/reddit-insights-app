# Reddit Insights Chatbot with RAG
## MSDS696 – Data Science Practicum II
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

### STEP 01- Clone the repository:
```bash
git clone https://github.com/jonishk/reddit-insights-app.git
cd reddit-insights-app
```
### STEP 02- Create and Prerequisites packages
```bash
conda env create -f environment.yml
```
### STEP 03- Environment Variables (.env file)
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

Estimated average run‑time per pipeline step (3 full runs)

| Step                | Run 1 | Run 2 | Run 3 | **Average** |
|---------------------|-------|-------|-------|-------------|
| Data Collection     | 0.97 min | 7.53 min | 5.40 min | **4.6 min** |
| Clean + Semantic    | 32.43 min| 28.87 min| 30.95 min| **30.8 min** |
| Sentiment Analysis  | 0.38 min | 0.33 min | 0.78 min | **0.5 min** |
| Indexing (Pinecone) | 0.22 min | 0.43 min | 0.97 min | **0.5 min** |
| **Full pipeline**   | 33.99 min| 37.18 min| 38.09 min| **≈36.4 min** |

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
----------------------------------------------------------------------------------------------------
## Quick start (Windows)

1. Clone the repo: `git clone https://github.com/jonishk/reddit-insights-app.git`
2. `cd reddit-insights-app`
3. Create the conda environment: `conda env create -f environment.yml`
4. Run the starter script: `start_app.bat`

## Quick start (macOS / Linux)

1. Clone the repo: `git clone https://github.com/jonishk/reddit-insights-app.git`
2. `cd reddit-insights-app`
3. Create the conda environment: `conda env create -f environment.yml`
4. Make the script executable: `chmod +x start_app.sh`
5. Run the starter script: `./start_app.sh`






