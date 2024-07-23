# dmv-chatbot
A chatbot using the RAG approach to provide support and resources for domestic violence cases.

# Domestic Violence Support Chatbot

## Overview

Welcome to the Domestic Violence Support Chatbot project. This AI-driven chatbot leverages advanced machine learning techniques to offer immediate assistance and resources for individuals facing domestic violence situations. The chatbot utilizes the Retrieval-Augmented Generation (RAG) approach for dynamic and contextually relevant responses.

## Key Features

- **RAG Approach**: Combines retrieval and generation models to provide accurate and context-aware responses.
- **Pinecone Vector Database**: Efficiently manages and retrieves vector embeddings for high-performance searches.
- **Text Embeddings (Ada 002)**: Uses the `text-embedding-ada-002` model to generate meaningful text embeddings.
- **GPT-4o Mini**: Employs `gpt-4o mini` for generating responses based on retrieved information.

## Usage
1. **Configure the Vector Database:**
   Set up Pinecone and ensure it is correctly configured in your environment.

2. **Load Data:**
   The chatbot uses `PyPDFLoader` to read and process PDFs. Ensure your PDF documents are correctly placed in the specified directory.

3. **Run the Chatbot:**
   Start the chatbot server:
   ```bash
   streamlit run dmv_bot.py

## Known Issues
The chatbot may produce hallucinated responses due to incomplete data preprocessing. Ongoing improvements are being made to enhance data accuracy and reliability.

