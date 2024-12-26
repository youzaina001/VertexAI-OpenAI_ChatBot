import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_community.document_loaders import CSVLoader  # Updated import
from langchain_openai import OpenAIEmbeddings

# Conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []


# Load the saved knowledge base
def load_knowledge_base():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set the 'OPENAI_API_KEY' environment variable or add it to a .env file.")

        st.info("Loading the pre-saved knowledge base...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Small embedding model
        docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.success("Knowledge base loaded successfully!")
        return docsearch
    except Exception as e:
        st.error(f"Error loading the knowledge base: {e}")
        return None

# Query the loaded knowledge base
def query_knowledge_base(docsearch, question):
    try:
        # Initialize the Chat model
        llm = ChatOpenAI(model="chatgpt-4o-latest", temperature=0.7)

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

    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    conversation_history = []
    prompt = query
    conversation_history.append({"role": "user", "content": prompt})
    data = {
        "model": "chatgpt-4o-latest", 
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
        # Load CSV file into a DataFrame
        # df = pd.read_csv(dataset_file)
        
        # Display dataset overview
        # st.write("**Dataset Overview:**")
        # st.dataframe(df.head())
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




