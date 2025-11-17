# 📰 RAG Research Assistant for Web Articles

A URL-based Retrieval-Augmented Generation (RAG) system for research, news analysis.
Users provide URLs → the system scrapes the content → embeds it using BGE-Large → stores it in FAISS → and answers questions using LLM.

## 🚀 Features

- 🔗 URL-based ingestion — Paste any article or research link, and the system automatically extracts the text.

- 🧩 Semantic chunking using RecursiveCharacterTextSplitter.

- 🧠 High-accuracy embeddings using BAAI/bge-large-en-v1.5.

- 📦 FAISS vectorstore with optional persistence.

- 🤖 Modern RAG pipeline using LangChain’s updated RetrievalQA.

- 📝 Question answering with sources — users can ask any question based on the articles.

- 🎨 Streamlit interface for easy interaction.
