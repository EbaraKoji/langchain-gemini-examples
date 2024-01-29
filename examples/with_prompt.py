import os
import sys

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

sys.path.append(os.path.abspath(os.pardir))
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-pro")

prompt = ChatPromptTemplate.from_messages([("system", "You are Professional AI Engineer."), ("user", "{input}")])

chain = prompt | llm

result: str = chain.invoke({
    "input":
    "Please explain on Transformers for web developers "
    "who don't have solid mathematical understandings of Deep Learning."
})

with open("results/with_prompt.txt", "wb") as f:
    f.write(bytes(result, 'utf-8'))
