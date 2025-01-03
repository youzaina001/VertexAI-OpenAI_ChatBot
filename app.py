import streamlit as st
import os
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI

client = OpenAI()
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel

# Load environment variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CSV_FILE_PATH = "Preprocessed_WS_DER_OTC_TOV_csv_col.csv"
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
INDEX_FILE_PKL = os.getenv("INDEX_FILE_PKL")
INDEX_FILE_FAISS = os.getenv("INDEX_FILE_FAISS")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 240))  # Default batch size if not set
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER")
SAVE_AS_FAISS = os.getenv("SAVE_AS_FAISS")

if not OPENAI_API_KEY:
    st.error("OpenAI API Key is missing. Set it as an environment variable.")
    st.stop()

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# Load the Knowledge Base
@st.cache_resource
def load_knowledge_base(provider=EMBEDDING_PROVIDER):
    try:
        # Select the embedding provider
        if provider == "openai":
            # OpenAI Embeddings
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            # Load the FAISS Index
            st.info(f"Loading FAISS index with {provider} embeddings...")
            docsearch = FAISS.load_local("openai", embeddings, allow_dangerous_deserialization=True)
            st.success("OpenAI Knowledge base loaded successfully!")
            return docsearch
        elif provider == "vertexai":
            # Initialize Vertex AI

            if not PROJECT_ID or not LOCATION:
                raise ValueError("Vertex AI Project ID or Location is missing. Set these in environment variables.")

            aiplatform.init(project=PROJECT_ID, location=LOCATION)

            # Vertex AI Embeddings
            model = TextGenerationModel.from_pretrained("text-bison")
            embeddings = lambda texts: [model.predict(text).embeddings for text in texts]

            # Load the FAISS Index
            st.info(f"Loading FAISS index with {provider} embeddings...")
            docsearch = FAISS.load_local("vertexai", embeddings, allow_dangerous_deserialization=True)
            st.success("Knowledge base loaded successfully!")
            return docsearch
        else:
            raise ValueError("Unsupported embedding provider. Use 'openai' or 'vertexai'.")

    except Exception as e:
        st.error(f"Failed to load knowledge base using {provider}: {e}")
        return None

# Query Knowledge Base
def query_knowledge_base(docsearch, query, k=3):
    try:
        results = docsearch.similarity_search_with_score(query, k=k)
        contexts = []
        for doc, score in results:
            contexts.append(doc.page_content)
        return contexts
    except Exception as e:
        return f"Error querying knowledge base: {e}"

# AI Query Processor
def ask_gpt(query, context):
    try:
        # Format the messages for the chat model
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]

        # Use the new ChatCompletion API
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=150)

        # Extract and return the assistant's reply
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with GPT-4 query: {e}"

# Initialize Vertex AI (with Llama 3.1 model)
def init_vertex_ai():
    if not PROJECT_ID or not LOCATION:
        raise ValueError("Vertex AI Project ID or Location is missing. Set these in environment variables.")

    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # Initialize Llama 3.1 Model
    llama_model = TextGenerationModel.from_pretrained("llama-3.1")
    return llama_model

# Query Knowledge Base using Vertex AI (Llama 3.1)
def ask_llama_3_1(query, context):
    try:
        llama_model = init_vertex_ai()

        # Construct prompt for Llama 3.1
        prompt = f"Context: {context}\n\nQuestion: {query}"

        # Query Llama 3.1 for response
        response = llama_model.predict(prompt)
        return response.text
    except Exception as e:
        return f"Error with Llama 3.1 query: {e}"

# Update AI Query Processor to Use Llama 3.1
def ask_ai(query, context):
    try:
        # Use Llama 3.1 model instead of GPT-4
        response = ask_llama_3_1(query, context)
        return response
    except Exception as e:
        return f"Error with AI query: {e}"

# Visualization Function
def generate_plot(df, query):
    try:
        # Infer plot type based on query keywords
        if "distribution" in query.lower() or "histogram" in query.lower():
            fig = px.histogram(df, x=df.columns[0], title="Histogram")
        elif "trend" in query.lower() or "line" in query.lower():
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Trend Analysis")
        elif "correlation" in query.lower() or "scatter" in query.lower():
            fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Correlation Analysis")
        else:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Bar Chart")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Plotting error: {e}")

# Streamlit App
st.title("AI Financial Assistant")
st.write("Upload a dataset and query insights using AI and knowledge base search.")

# Load Knowledge Base
docsearch = load_knowledge_base()
if not docsearch:
    st.error("Knowledge base could not be loaded.")
    st.stop()

# Dataset Upload
data_location = st.text_input("CSV File Location:", "")
if data_location and os.path.exists(data_location):
    df = pd.read_csv(data_location)
    st.write("**Dataset Preview:**")
    st.dataframe(df.head(10))

    # User Query Input
    user_query = st.text_area("Enter your query:")
    if st.button("Search Knowledge Base"):
        if user_query.strip():
            # Search Knowledge Base
            context_results = query_knowledge_base(docsearch, user_query)
            context = " ".join(context_results)

            # AI Insights
            st.write("**AI-Generated Insights:**")
            response = ask_gpt(user_query, context)
            st.text(response)

    # Visualization Generation
    if st.button("Generate Visualization"):
        if df is not None:
            st.write("**Data Visualization:**")
            generate_plot(df, user_query)
        else:
            st.error("Please upload a valid dataset.")
else:
    st.warning("Please provide a valid CSV file location.")
