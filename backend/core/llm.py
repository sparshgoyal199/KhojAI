from mistralai.client import Mistral
from dotenv import load_dotenv
import os
from groq import AsyncGroq
from langchain_groq import ChatGroq

load_dotenv()  # Load environment variables from .env file

groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)
from langchain_mistralai import ChatMistralAI

# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0
# )