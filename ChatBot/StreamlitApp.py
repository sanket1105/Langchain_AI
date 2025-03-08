import os

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OpenAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to user queries.",
        ),
        ("user", "Question: {question}"),
    ]
)


def generate_response(question, api_key, engine, temperature, max_tokens):
    openai.api_key = api_key

    llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


def main():
    # Title of the app
    st.title("Enhanced Q&A Chatbot With OpenAI")

    # Sidebar for settings
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

    # Select the OpenAI model
    engine = st.sidebar.selectbox(
        "Select OpenAI model", ["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo"]
    )

    # Adjust response parameters
    temperature = st.sidebar.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.7
    )
    max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=200, value=150)

    # Main interface for user input
    st.write("Go ahead and ask any question")
    user_input = st.text_input("You:")

    if user_input and api_key:
        response = generate_response(
            user_input, api_key, engine, temperature, max_tokens
        )
        st.write("Assistant:", response)

    elif user_input:
        st.warning("Please enter your OpenAI API key in the sidebar")
    else:
        st.write("Please enter your question above")


if __name__ == "__main__":
    main()
