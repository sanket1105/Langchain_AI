import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()

##create chain
chain = prompt_template | model | parser


## App definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces",
)

## Adding chain routes
# This adds API routes to our FastAPI app for the translation chain
# The add_routes function creates:
# - POST /chain/invoke - Main endpoint to run the chain
# - GET /chain/input_schema - Returns JSON schema for the expected input
# - GET /chain/output_schema - Returns JSON schema for the output
# - GET /chain/config_schema - Returns config schema if any
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
