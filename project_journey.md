# Project Journey – Reddit Insights Chatbot with RAG

##  Who
I’m **Jonish Bishwakarma**, a graduate student in the **Master of Science in Data Science (MSDS)** program at Regis University.  
This project was completed as part of **MSDS 692 – Data Science Practicum I**, in collaboration with **MSP Shift**, a managed services provider introduced to me by my professor.  

MSP Shift focuses on helping organizations select and integrate software tools efficiently.  
They proposed the idea for this project to build a **research chatbot** capable of uncovering software trends, user pain points, and discussions from Reddit to help guide their decision-making and client support.

---

## What
The goal of this project was to **build a Retrieval-Augmented Generation (RAG) chatbot** that surfaces insights from Reddit discussions about software tools used in specific industries — **law, construction, and technology**.  

The system collects, cleans, and analyzes Reddit data, then allows users to **query insights conversationally** through a web-based chatbot interface.  

The chatbot retrieves relevant Reddit posts using **semantic search (via Pinecone)** and generates **context-grounded summaries** using **GPT-3.5-turbo**.  
To assess performance, I also compared the **RAG model** against a **standard LLM-only approach**.

---

## Why
MSP Shift often helps clients choose between similar software products.  
With so many competing platforms and features, the team wanted a tool that could **analyze real user feedback from the web** instead of relying solely on marketing materials.

Reddit is a goldmine for this kind of qualitative data and professionals share what tools they use, what they like, and what frustrates them.  
The challenge, however, is that Reddit data is unstructured and noisy.  
This project bridges that gap by turning those unstructured discussions into **organized, searchable, and summarized insights**.

---

## When
This project was developed during the **Fall 2025 term** (Weeks 1–8) as part of my Practicum I coursework.

| Week | Focus Area |
|------|-------------|
| 1 | Project Definition and Proposal |
| 2 | Reddit Data Collection |
| 3 | Data Cleaning and Preprocessing |
| 4 | Keyword Extraction and Software Mention Mapping |
| 5 | Sentiment Analysis and Pain Point Identification |
| 6 | Building and Testing the RAG + LLM Pipelines |
| 7 | Model Evaluation and Refinement |
| 8 | Final Documentation and Presentation |

---

## How (Project Workflow)

### **1. Data Collection**
- Gathered **57,497 Reddit posts** from subreddits such as `r/LegalTech`, `r/Construction`, and `r/Technology`.
- Used the **Pushshift API** to extract post text, metadata, and engagement data.

### **2. Data Cleaning & Preprocessing**
- Removed duplicates, empty posts, and irrelevant content.
- Filtered to **1,378 high-quality posts**.
- Added **domain labels** (Law, Construction, Tech) and extracted **software mentions** and **keywords**.

### **3. Sentiment Analysis**
- Applied the **VADER sentiment analyzer** to each post.
- Classified user attitudes toward software as positive, neutral, or negative.
- Found common themes such as satisfaction, pricing frustration, or workflow complexity.

### **4. RAG Chatbot Development**
- Used **LangChain**, **Pinecone**, and **HuggingFace embeddings** .
- Created semantic vector embeddings for all posts and stored them in Pinecone.
- Connected **GPT-3.5-turbo** to generate natural-language answers grounded in retrieved Reddit content.

### **5. Chatbot Interface**
- Built a **Flask-based web app** with a modern UI for interactive querying.
- Added buttons to run each pipeline stage: data collection, cleaning, sentiment analysis, indexing, and evaluation.
- Integrated **live log streaming (SSE)** to display real-time progress updates.

### **6. Evaluation**
- Created a test set of **20 questions** (15 in-scope, 5 out-of-scope).
- Compared **RAG vs LLM-only** responses.
- Used metrics such as **Accuracy**, **Precision**, **Recall**, and **F1 Score**.

| Metric | RAG | LLM |
|---------|-----|-----|
| Accuracy | 0.65 | 1.00 |
| Precision | 1.00 | 1.00 |
| Recall | 0.53 | 1.00 |
| F1 Score | 0.70 | 1.00 |

> *Interpretation:*  
> RAG provided highly accurate and grounded answers, but sometimes chose not to answer when data wasn’t available — leading to lower recall but higher factual reliability.  
> LLM-only appeared perfect statistically but often responded confidently without evidence, highlighting the importance of RAG grounding.

---

## Roadblocks & How I Overcame Them

| Challenge | Solution |
|------------|-----------|
| **Long scraping times caused browser timeouts.** | Added real-time streaming logs via Server-Sent Events (SSE). |
| **Initial RAG responses were too generic.** | Improved embedding filters using domain keywords (Law, Construction). |
| **Pinecone index not created before app startup.** | Automated index creation within the indexing script. |
| **Perfect evaluation scores (1.00) — unrealistic.** | Refined scoring logic to measure relevance more accurately. |
| **Managing complex logs from multiple scripts.** | Implemented modular pipeline execution and progress visualization. |

---

## Key Findings

- **Law:** Tools like *Clio* and *Filevine* dominated mentions.  
- **Construction:** Professionals discussed *Procore*, *Autodesk*, and *Microsoft Project*, with many pain points around scheduling coordination.  
- **Tech:** *Microsoft Teams* and *Outlook* were top collaboration tools.  

Overall, RAG produced grounded, data-backed responses that helped summarize thousands of Reddit opinions into actionable insights.

---

## Lessons Learned

- **Data quality > data quantity:** Well-cleaned and labeled text improved accuracy.  
- **RAG prioritizes truth over coverage:** It avoids hallucination by refusing to answer unsupported questions.  
- **User-centered design matters:** Building an interactive interface helped connect technical work to real use cases.  
- **Evaluation isn’t just about numbers:** A 1.00 accuracy score doesn’t always mean the model is right — interpretability matters.  

---

## Future Work

- Expand dataset to include **more subreddits and industries**.  
- Add **topic clustering and trend detection** to identify emerging tools.  
- Deploy chatbot online for **public demo and user testing**.  
- Explore **fine-tuned embeddings** or domain-adapted LLMs for higher recall.  

---

## Personal Reflection

This project was a turning point in my data science journey.  
It taught me not just how to build AI systems, but how to make them **useful in real business contexts**.  

Working with MSP Shift made this project feel real — like something that could directly support a company’s decision-making process.  
It also gave me experience balancing **academic rigor with practical goals**.

From handling massive data to debugging pipeline dependencies, I learned how to manage an end-to-end AI project independently from concept to deployment.  

Overall, this project combined everything I’ve learned so far:  
Data wrangling, NLP, machine learning, evaluation, and user-focused design and all tied together through a meaningful real-world problem.

---

## Visual Summary

> **Reddit Discussions → Data Pipeline → Sentiment Analysis → RAG Chatbot → Insights**

<p align="center">
  <img src="static/Screenshot 2025-10-13 005827.png" width="500" alt="Chatbot Interface Screenshot"/>
</p>

---

## Acknowledgements

- **Professor Christy Pearson**, for mentorship and continuous feedback  
- **MSP Shift**, for providing the real-world context and initial project vision  
- **LangChain**, **Pinecone**, **HuggingFace**, and **OpenAI**, for the powerful open-source tools that made this project possible  

---

*Thank you for exploring my project! This repository contains all code, data pipeline scripts, and evaluation results for the Reddit Insights Chatbot with RAG.*
