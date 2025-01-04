import streamlit as st
import os
import logging
import requests
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
from vertexai.language_models import TextGenerationModel,TextEmbeddingModel

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
            model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            embeddings = lambda texts: [embedding.values for embedding in model.get_embeddings(texts if isinstance(texts, list) else [texts])]

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

# Query the loaded knowledge base
def query_knowledge_base(docsearch, question):
    try:
        # Initialize the Chat model
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # Search the knowledge base for similar content
        result = docsearch.similarity_search_with_score(question, k=1)

        if result and len(result) > 0:
            doc_content, score = result[0]

            # Construct a prompt with context and question
            prompt = f"You are a helpful assistant. Based on this context: '{doc_content.page_content}', answer the question: {question}"

            # Generate a response using the chat model
            response = llm.predict(prompt)
            return response  # Return the generated response
        else:
            return "No relevant information found in the knowledge base."
    except Exception as e:
        return f"Error querying the knowledge base: {e}"

def ask_gpt(query):

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    conversation_history = []
    prompt = query
    conversation_history.append({"role": "user", "content": prompt})
    data = {
        "model": "gpt-4o-mini", 
        "messages": conversation_history,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_json = response.json()
        message_content = response_json['choices'][0]['message']['content']
        return message_content
    else:
        return f"Error: {response.status_code}, {response.text}"

# Streamlit app layout
st.title("Your financial report assistant")
# Sidebar options for navigation
#option = st.sidebar.selectbox("Choose an option", ("Chat with GPT", "Generate Insights and Plot"))


st.write("Generate dataviz and financial reports in a simple way by prompting")
docsearch = load_knowledge_base()
st.write("Upload a dataset (CSV) to analyze.")

# Upload CSV
# dataset_file = st.file_uploader("Upload CSV file", type="csv")

# User input

data_location = st.text_input("Your csv location:", "",)

if os.path.exists(data_location):
    df = pd.read_csv(data_location)
    cols = ", ".join(df.columns)
    csv_text = df.head(100).to_csv(index=False)
    
    try:
        TemplateBegin = '''
        Your are a great statistician which have great skill in
        data analysis and data visualation. Your python programming skills is good outstanding
        '''

        # Ask GPT to generate insights from the dataset
        cols = ", ".join(df.columns)
        csv_text = df.head(20).to_csv(index=False)

        # head_query = """Make an analysis like a data scientist 
        # """
        user_query = st.text_area("Ask a question about the dataset:")
        # user_query += head_query
        if st.button("Ask GPT"):
            if user_query.strip():
                with st.spinner("Generating insights..."):
                    bot_response = query_knowledge_base(docsearch, user_query)
                st.write(f"**Bot's Insights:** {bot_response}")
            else:
                st.error("Please enter a question about the dataset!")


        seaborn_theme = """ sns.set_theme(
            style = 'whitegrid', 
            palette = 'colorblind', 
            font = 'Arial', 
            rc = {
            'axes.spines.top': False, 
            'axes.spines.right': False, 
            'axes.spines.left': False,             
            'grid.linestyle': '-.',
            'text.color': '#010035',
            'font.family': 'Arial',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'grid.color': '#f0f0f0'
            }
        )"""

        # Dashbord
        head_query_plot = f"""Here are the columns of the file of my csv: {cols}
        Here is the 20 first lines of my csv
        {csv_text}

        The location of my csv is {data_location}
        """

        TemplateEnd = f"""(Generate A full python Code (use pandas to read a csv (and matplotlib as plt if needed)),
        use seaborn with this theme {seaborn_theme} to create a pertinent dashbord according to the data,
        you can use multiple type of garphics to create a beautiful and readable dashbord.
        Choose pertinent graphics according to the data
        Return only the python code.
        Give a return that can be run directly with exec and lanchable in streamlit
        Make sure that the code is runnable
        """
        plot_query = TemplateBegin + head_query_plot + user_query + TemplateEnd
        # Example plot (you can modify this part to plot specific columns or analyses)
        if st.button("Generate Plot"):
            if plot_query.strip():
                with st.spinner("Generating insights..."):
                    response = ask_gpt(plot_query)
                    # st.write(response)
                    r = response.split("```")[1].replace("python", "")
                    exec(r)
            
            else:
                st.error("A.")
    except Exception as e:
        st.error(f"An error occurred: {e}")