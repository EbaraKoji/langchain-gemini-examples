import os
import sys

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

sys.path.append(os.path.abspath(os.pardir))

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-pro", max_output_tokens=800)
result = llm.invoke("LLMについて説明してください。")

with open("results/japanese_question.txt", "wb") as f:
    f.write(bytes(result, "utf-8"))
