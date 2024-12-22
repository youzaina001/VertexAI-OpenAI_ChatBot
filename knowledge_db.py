import os
import argparse
import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.document_loaders import CSVLoader  # Updated import
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Path to the CSV file
CSV_FILE_PATH = "Preprocessed_WS_DER_OTC_TOV_csv_col.csv"

api_key = os.getenv("OPENAI_API_KEY")

def initialize_knowledge_base(file_path, api_key):
    try:
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please provide a valid API key.")

        os.environ["OPENAI_API_KEY"] = api_key

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file '{file_path}' not found.")

        print("Loading documents from the CSV file...")
        # Optional preprocessing step for large files
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} rows from the CSV file.")

        # Load documents
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()

        print("Initializing the embedding model...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Small embedding model

        print("Creating FAISS index...")
        docsearch = FAISS.from_documents(documents, embeddings)

        # Save the FAISS index to disk
        save_path = "faiss_index"
        docsearch.save_local(save_path)
        print(f"Knowledge base initialized and saved to '{save_path}' successfully!")
    except Exception as e:
        print(f"Error initializing the knowledge base: {e}")

if __name__ == "__main__":
    # Call the initialization function
    initialize_knowledge_base(CSV_FILE_PATH, api_key)
