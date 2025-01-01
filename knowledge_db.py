import os
import time
import numpy as np
import pickle
import logging
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
import faiss  # FAISS for similarity search

# OpenAI and Vertex AI Imports
from langchain_community.embeddings import OpenAIEmbeddings
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Environment Variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Constants
CSV_FILE_PATH = "Preprocessed_WS_DER_OTC_TOV_csv_col.csv"
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
INDEX_FILE_PKL = os.getenv("INDEX_FILE_PKL")
INDEX_FILE_FAISS = os.getenv("INDEX_FILE_FAISS")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 240))  # Default batch size if not set
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
SAVE_AS_FAISS = os.getenv("SAVE_AS_FAISS")

# Initialize Vertex AI (only if using Vertex AI embeddings)
if EMBEDDING_PROVIDER == "vertexai":
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Load CSV Data
logging.info("Loading CSV data...")
data = pd.read_csv(CSV_FILE_PATH)

# Combine Columns into Context
logging.info("Preparing data context...")
data['context'] = data.apply(lambda x: f"{x['Frequency']} {x['Measure']} {x['Instrument']} {x['Risk category']} "
                                        f"{x['Reporting country']} {x['Counterparty sector']} "
                                        f"{x['Counterparty country']} {x['Underlying risk sector']} "
                                        f"{x['Currency leg 1']} {x['Currency leg 2']}", axis=1)

# Embedding Generator
def generate_embeddings(data, provider, batch_size=BATCH_SIZE):
    """
    Generate embeddings using the specified provider: 'openai' or 'vertexai'.
    """
    logging.info(f"Using {provider} embeddings...")
    all_embeddings = []
    contexts = data['context'].tolist()

    if provider == "vertexai":
        # Load Vertex AI Model
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")

        # Process data in batches
        for start in range(0, len(contexts), batch_size):
            end = start + batch_size
            batch = contexts[start:end]
            logging.info(f"Processing batch {start + 1} to {end}...")

            # Generate embeddings
            response = model.get_embeddings(batch)
            time.sleep(10)  # Sleep to avoid rate limits
            batch_embeddings = [res.values for res in response]
            all_embeddings.extend(batch_embeddings)

    elif provider == "openai":
        # Load OpenAI Model
        api_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

        # Process data in batches
        for start in range(0, len(contexts), batch_size):
            end = start + batch_size
            batch = contexts[start:end]
            logging.info(f"Processing batch {start + 1} to {end}...")

            # Generate embeddings
            batch_embeddings = embeddings.embed_documents(batch)
            time.sleep(5)  # Sleep to avoid rate limits
            all_embeddings.extend(batch_embeddings)

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    return np.array(all_embeddings)

# Generate Embeddings
logging.info("Generating embeddings...")
embeddings = generate_embeddings(data, EMBEDDING_PROVIDER)
logging.info(f"Embeddings generated successfully! Shape: {embeddings.shape}")

# Create FAISS Index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

if SAVE_AS_FAISS:
    # Combine file path using pathlib
    index_file_path_faiss = Path(EMBEDDING_PROVIDER) / INDEX_FILE_FAISS

    # Ensure the directory exists
    index_file_path_faiss.parent.mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist

    # Save index and metadata separately
    faiss.write_index(index, str(index_file_path_faiss))  # Save binary index file

    # Save metadata
    data.to_csv(str(Path(EMBEDDING_PROVIDER) / "financial_metadata.csv"), index=False)  # Save metadata as CSV   

    logging.info(f"FAISS index saved as {INDEX_FILE_FAISS} and metadata saved as 'financial_metadata.csv'.")
else:
    # Combine file path using pathlib
    index_file_path_pkl = Path(EMBEDDING_PROVIDER) / INDEX_FILE_PKL

    # Save index and data together using pickle
    with open(index_file_path_pkl, 'wb') as f:
        pickle.dump({'index': index, 'data': data}, f)

    logging.info(f"FAISS index saved as {INDEX_FILE_PKL}")