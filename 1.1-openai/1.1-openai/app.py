import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Set page config
st.set_page_config(page_title="Q&A Bot", page_icon="🤖")

# Initialize OpenAI model and conversation chain
if "chain" not in st.session_state:
    # Use OpenAI model
    # temperature=0.7 controls randomness in responses
    llm = OpenAI(temperature=0.7, model="gpt-3.5-turbo-instruct")

    st.session_state.chain = ConversationChain(
        llm=llm, memory=ConversationBufferMemory()
    )

# App title
st.title("AI Q&A Assistant")
st.write("Ask me anything!")

# Get user input
user_input = st.text_input("Your question:", key="input")

# Generate response
if user_input:
    try:
        response = st.session_state.chain.run(user_input)

        # Display response
        st.write("AI Response:")
        st.write(response)
    except Exception as e:
        st.error("An error occurred. Please try again.")
