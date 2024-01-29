import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.abspath(os.pardir))
load_dotenv()

# gemini does not support system messages!
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, max_output_tokens=200)

prompt = ChatPromptTemplate.from_messages([("system", "You are Professional AI Engineer."), ("user", "{input}")])

chain = prompt | llm

result: AIMessage = chain.invoke({"input": "Please explain who are you in a brief sentence."})

with open("results/chat_with_prompt.txt", "wb") as f:
    f.write(bytes(result.content, "utf-8"))
