# RAG Relational Intelligence System
##  AI-Powered SQL Insights with LLMs, Retrieval-Augmented Generation & Beautiful Visualizations

📌 Overview

RAG Relational Intelligence System is an intelligent analytics tool that lets users ask natural-language questions about relational datasets and receive:
	•	📊 Exact, deterministic SQL-style answers
	•	📈 Auto-generated visualizations (bar charts, pie charts, treemaps, line graphs)
	•	🧠 LLM-powered summaries that explain results simply
	•	🔍 RAG-style retrieval of relevant items using fuzzy search
	•	⚡ Fast, secure, self-contained local processing (no backend required on cloud)

This project demonstrates how RAG + LLMs + relational databases can be combined to build a smart analytics assistant / SQL Copilot system — perfect for Data Science, AI Engineering, ML-Ops, and Applied ML portfolios.
🚀 Features

✅ Natural Language Question Answering

Ask questions like:
	•	“Which products appear most frequently in prior orders?”
	•	“List items containing the word apple.”
	•	“Show least ordered products.”
	•	“How many orders happen on each day of the week?”

The system automatically detects intent (aggregation vs retrieval).

⸻

🧮 Deterministic Local Query Engine (No FastAPI Needed)
	•	Performs exact Pandas aggregations
	•	Offers SQL-like operations: count, sum, average, top-k, grouping
	•	Works 100% offline inside Streamlit Cloud

⸻

🧠 LLM-Enhanced Summaries (With HuggingFace Llama-3)

Uses HuggingFace Inference API to generate:
	•	Friendly summaries
	•	Insights in plain English
	•	Follow-up questions
	•	Optional chart suggestions

If LLM output is wrong, contradictory, or empty → a fallback deterministic summary is generated.

⸻

 # 📊 Rich Visualizations

Auto-generated:
	•	Bar charts
	•	Pie charts
	•	Treemaps
	•	Line charts

Built using Plotly Express for interactive visual insights.

⸻

🔎 Fuzzy Product Retrieval (RAG-like search)

Extracts relevant items using token-level search & normalization.
Example:
	•	Input: “apple”
	•	Returns: all related products + LLM summary

⸻

 # 🌐 Streamlit Cloud Deployment

Runs smoothly on Streamlit Cloud without FastAPI.
Your live app: (Insert your Streamlit cloud link here)
 # 🏗️ Tech Stack
 Component
Technology
Frontend
Streamlit
Processing
Pandas, NumPy
Visualization
Plotly Express
LLM
Meta-Llama-3-8B-Instruct (via HuggingFace Inference API)
RAG-Style Retrieval
Custom fuzzy search
Dataset
Instacart cart analysis CSVs
# 📂 Project Structure
rag-streamlit-deploy-project/
│
├── streamlit_app.py
├── streamlit_app_experiment.py
├── data/
│   └── data/
│       └── instacart/
│           ├── products.csv
│           ├── aisles.csv
│           ├── departments.csv
│           ├── orders.csv
│           └── order_products__prior.csv
└── requirements.txt
