from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
from deepseek import DeepSeekAPI as Deepseek
import streamlit as st
import pandas as pd
import os

# fetch secrets from streamlit app
deepseek_api_key =  st.secrets['DEEPSEEK_API_KEY']
model_name = st.secrets['MODEL_NAME']
openai_api_base = st.secrets['OPENAI_API_BASE']


# Setup your Langchain model
llm = ChatOpenAI(
    model_name= model_name,
    openai_api_key= deepseek_api_key,
    openai_api_base =   openai_api_base,
    temperature = 0
    )  


# Load your dataset (replace with your dataset file)
dataset = pd.read_csv('wildfire_subset.csv')

pandas_agent = create_pandas_dataframe_agent(
    llm,
    dataset, 
    verbose = True,
    agent_type="openai-tools",
    handle_parsing_errors = True,
    allow_dangerous_code= True
    )

@tool
def query_dataset(query: str):
    """
    This is a function to query from the dataset
    """
    combined = dataset.astype(str).apply(lambda row: ''.join(row.values), axis = 1)

    mask = combined.str.contains(query, case=False, na=False)
    results = dataset[mask]

    return results.head().to_string()

tools = [query_dataset]

WILDFIRE_KEYWORDS = ["general cause","wildfire", "fire", "fires", "class", "burn", "forest", "smoke", "blaze", "emissions", "evacuation"]

def is_relevant_question(query: str) -> bool:
    return any(word in query.lower() for word in WILDFIRE_KEYWORDS)

# Set up the Streamlit interface
st.set_page_config(layout="wide")


st.title('Alberta Wildfire Information Chatbot 🔥')
st.write('This chatbot provides information on the wildfires between 2006 and 2023 in the province of Alberta. Created by Archana Senthil')

st.markdown(
    """
    <div class="header">
        <span><strong>My Profile:</strong></span>
        🔗 <a href="https://www.linkedin.com/in/archana-senthil/" target="_blank">LinkedIn</a>
        🐙 <a href="https://github.com/i-archanasenthil" target="_blank">GitHub</a>
        📄 <a href="assets/resume.pdf" download="Archana-Senthil--Resume.pdf">Download Resume</a>  
    </div>
    """,
    unsafe_allow_html=True
)

if 'history' not in st.session_state:
    st.session_state['history'] = []

for message in st.session_state['history']:
    if message['role'] == 'user':
        st.chat_message("user").write(message['content'])
    else:
        st.chat_message("assistant").write(message['content'])

user_input = st.chat_input("Ask me anything:")

if user_input:
    # Get the response from the Langchain agent
    try:
        st.session_state['history'].append({'role':'user','content': user_input})
        st.chat_message("user").write(user_input)
        if is_relevant_question(user_input):
            response = pandas_agent.run(user_input)
        else:
            response = "Sorry not within my scope, but I will gladly answer any questions about the wildfire from 2006 to 2024"
        st.session_state['history'].append({'role':'assistant','content': response})
        st.chat_message("assistant").write(response)
    except Exception as e:
        st.error("Something went wrong")

