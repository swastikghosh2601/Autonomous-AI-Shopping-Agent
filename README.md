# Autonomous E-Commerce Personal Shopping Agent

An AI-powered personal shopping assistant that understands user preferences, retrieves relevant products using semantic search, and delivers personalized recommendations through a conversational interface.


# 📌 Project Overview

Modern e-commerce platforms overwhelm users with thousands of product choices. Traditional recommendation systems rely heavily on collaborative filtering and fail to capture user intent, preferences, and context effectively.

This project proposes an Autonomous Personal Shopping Agent that combines:

Semantic product retrieval (RAG)

User preference modeling

Multimodal understanding (text + vision)

Reinforcement learning–based optimization

Conversational AI

The system behaves like a human personal shopper — understanding what the user wants, browsing products intelligently, and recommending the most suitable options.


# 🎯 Objectives

Build a semantic product search system using vector embeddings

Implement Retrieval-Augmented Generation (RAG) for accurate recommendations

Model dynamic user preferences from interactions

Optimize recommendations using reinforcement learning principles

Provide a conversational shopping experience

Simulate price negotiation and offers


# 🧠 Key Features

🔍 Semantic product search using vector databases

🧠 Personalized recommendations based on user behavior

💬 Conversational shopping interface

📸 Vision + text–based preference learning

🎯 RL-based recommendation optimization

💸 Simulated price negotiation logic


# 🏗️ System Architecture
User Interface (Chat UI)
        ↓
Conversational AI (LLM)
        ↓
RAG Product Retriever (FAISS / Chroma)
        ↓
Recommendation Engine (ML + RL)
        ↓
User Preference Store (MongoDB)

# 🛠️ Tech Stack
Frontend

React / Next.js

Chat-based UI

Backend

FastAPI (Python)

REST APIs

AI / ML

Sentence Transformers (text embeddings)

CLIP / Vision Transformers (image embeddings)

Large Language Models (LLM)

Contextual Bandits / Reinforcement Learning

Databases

MongoDB (user data, logs)

FAISS / Chroma (vector database)

# 📂 Project Structure
autonomous-shopping-agent/
├── data/
│   └── products.csv
├── embeddings/
│   ├── product_vectors.npy
│   └── product_index.faiss
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── embed.py
│   ├── recommender.py
│   └── models.py
├── notebooks/
│   └── experiments.ipynb
├── requirements.txt
├── README.md
└── report/
    └── final_project_report.pdf

# 🚀 Implementation Phases
Phase 1: Product Catalog + RAG

Dataset collection and cleaning

Product embedding generation

Vector database creation

Semantic product retrieval API

Phase 2: User Preference Modeling

User interaction logging

Preference vector construction

Cold-start handling

Phase 3: Multimodal Learning

Image embedding with Vision Transformers

Fusion of text and image preferences

Phase 4: Recommendation Optimization

Contextual multi-armed bandits

Reward-based recommendation updates

Phase 5: Conversational Agent

Intent extraction

Explainable recommendations

Context-aware responses

Phase 6: Price Negotiation (Simulated)

Rule-based discount logic

Loyalty and cart-based offers


# 📊 Evaluation Metrics

Precision@K

Recall@K

Click-through rate (CTR)

Conversion rate (simulated)

Recommendation diversity


# 🧪 Example Query
User: Suggest budget sneakers under ₹3000
Agent: Based on your preference for lightweight footwear and budget constraints,
        here are the top recommendations...


# 📈 Results

Accurate semantic retrieval of products

Improved personalization over static recommenders

Human-like conversational shopping flow

Demonstrates full AI product lifecycle


# ⚠️ Challenges & Considerations

Cold-start problem for new users

Feedback loops and over-personalization

Privacy and ethical handling of user data

Computational constraints for large datasets


# 🎓 Academic Relevance

This project demonstrates applied knowledge of:

Machine Learning

Natural Language Processing

Computer Vision

Reinforcement Learning

Software Engineering

AI Product Design


#📜 License

This project is developed for academic and learning purposes only.


#👨‍💻 Author

Swastik Ghosh
BTECH in Computer Science and Business Systems
AI / Machine Learning Enthusiast
